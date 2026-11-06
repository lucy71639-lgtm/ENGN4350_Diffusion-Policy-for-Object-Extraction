"""
LiftToolWedged: a robosuite environment based on Lift.

Setup: a thin aluminium "tool" bridges two fixed stands. Each stand is a
rectangular prism with a pair of fixed cubes on top, one cube on each side
of the tool, pinching it (like the silicone holders in the extraction
force diagram: Fa vs. Fr + mg).

    Front view (x -> right, z up):

         |<-3cm->|<---- 4 cm ---->|<-3cm->|
         +-------------------------------+
         |            TOOL               |   12 x 0.5 x 1 cm (free body)
         +-------------------------------+
            +--+                  +--+
            |  |  prism (fixed)   |  |       1 x 4 x 2 cm (x, y, z)
            +--+                  +--+

    Top view of ONE stand (x -> right, y up):

            +---+   cube_?_pos (fixed, 1 cm)
         ===|   |=== <- tool passes between the cubes (along x)
            +---+   cube_?_neg (fixed, 1 cm)

Objects
    - prism_l / prism_r              : fixed, 1 x 4 x 2 cm, sit on the table.
    - cube_l_neg, cube_l_pos         : fixed 1 cm cubes on prism_l (red).
    - cube_r_neg, cube_r_pos         : fixed 1 cm cubes on prism_r (blue).
    - tool                           : free joint, 12 x 0.5 x 1 cm aluminium
                                       (density 2700 kg/m^3 -> ~16.2 g),
                                       rests on the two prism tops.

Design notes (same conventions as LiftTightCube)
  * Fixed objects are created with `joints=[]`, so robosuite welds them to
    the worldbody. They can't be moved with `set_joint_qpos`, so their
    position is baked into the MJCF `pos` attribute in `_load_model`.
  * Only the tool is registered with the placement sampler, and it is held
    at x = y = 0, rotation = 0 (the pinch geometry leaves no room for
    randomisation). The sampler drops it from `z_offset` above the prism
    tops, so it settles onto them in the first few steps.
  * `gap` is the signed clearance between each cube's inner face and the
    tool's side face. Negative = interference (pre-load). With the default
    -0.0005 each cube overlaps the tool by 0.5 mm per side; MuJoCo resolves
    this as a soft contact. The resulting clamp force depends on the contact
    solver parameters (solref/solimp), not on a physical stiffness.
  * `friction` is applied to every geom. MuJoCo combines contact friction
    with the componentwise MAX of the two geoms, so this value directly sets
    tool-vs-cube and tool-vs-prism friction.
  * Success = tool centre is `lift_success_height` (default 5 cm) above its
    initial resting height.

Tested against the robosuite 1.4.x / 1.5.x API shape (same assumptions as
the LiftTightCube file). If the texture "SteelScratched" does not exist in
your install, swap it for any other texture name from robosuite's TEXTURES.
"""

import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
import robosuite.utils.transform_utils as T


class LiftToolWedged(Lift):
    """
    Lift task variant: a free aluminium tool pinched between fixed cubes,
    bridging two fixed prisms.

    Args:
        gap (float): signed clearance (m) between each cube's inner face and
            the tool's side face. 0 = just touching, negative = interference
            (pre-load), positive = clearance.
        tool_size (3-tuple): FULL extents (m) of the tool: (length along x,
            width along y, height along z).
        cube_half_size (float): half-extent (m) of the fixed cubes
            (0.005 -> 1 cm cubes).
        prism_size (3-tuple): FULL extents (m) of each fixed prism:
            (x width, y length, z height).
        tool_overhang (float): how far (m) the tool extends past the outer
            edge of each prism along x. Prism x-positions are derived from
            this so the tool is centred between them.
        tool_density (float): kg/m^3. 2700 = aluminium.
        friction (3-tuple): (sliding, torsional, rolling) friction applied
            to all geoms.
        lift_success_height (float): lift (m) above the initial resting
            height of the tool at which the task counts as solved.
        **kwargs: forwarded to Lift / ManipulationEnv (robots,
            controller_configs, has_renderer, etc.)
    """
    # ------------------------------------------------------------------
    # init default variables are what the policy was trained on.
    def __init__(
        self,
        gap=-0.00075, #Trained on -0.00075
        tool_size=(0.16, 0.005, 0.025),
        cube_half_size=0.005,
        prism_size=(0.025, 0.06, 0.025),
        tool_overhang=0.03,
        tool_density=2000.0, #slightly lower than default. Default of aluminium is 2700
        friction=(0.5, 0.005, 0.0001),
        lift_success_height=0.1,
        **kwargs,
    ):
        self.gap = gap
        self.tool_size = tuple(tool_size)
        self.cube_half_size = cube_half_size
        self.prism_size = tuple(prism_size)
        self.tool_overhang = tool_overhang
        self.tool_density = tool_density
        self.contact_friction = friction
        self.lift_success_height = lift_success_height

        # approximate tool mass (kg); aluminium default -> ~0.0162 kg
        self.tool_mass = tool_density * float(np.prod(self.tool_size))

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_material(tex_name):
        return CustomMaterial(
            texture=tex_name,
            tex_name=tex_name.lower(),
            mat_name=f"{tex_name.lower()}_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

    def _make_box(self, name, half_size, rgba, tex_name, fixed, density=None):
        """BoxObject with a free joint (fixed=False) or welded to the world (fixed=True)."""
        kwargs = dict(
            name=name,
            size_min=list(half_size),
            size_max=list(half_size),
            rgba=rgba,
            material=self._make_material(tex_name),
            friction=self.contact_friction,
        )
        if fixed:
            kwargs["joints"] = []  # no joint -> rigidly welded to worldbody
        if density is not None:
            kwargs["density"] = density
        return BoxObject(**kwargs)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _load_model(self):
        # Skip Lift._load_model (hardcodes a single cube); redo the same
        # arena/robot setup ourselves.
        super(Lift, self)._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # ---------------- geometry (all in metres) ------------------------
        tx, ty, tz = self.tool_size
        hx, hy, hz = tx / 2, ty / 2, tz / 2
        c = self.cube_half_size
        px, py, pz = self.prism_size
        ox, oy, base = self.table_offset  # table surface height = base

        # prism centre x so that `tool_overhang` of tool sticks out past the
        # prism's OUTER edge:  hx = x_prism + px/2 + overhang
        x_prism = hx - self.tool_overhang - px / 2
        # cube centre y: inner face sits `gap` away from the tool's side face
        y_cube = hy + self.gap + c

        assert x_prism > 0, "tool_overhang too large: prisms would overlap/cross"
        assert y_cube + c <= py / 2 + 1e-9, "cubes do not fit on the prism top (increase prism y length)"

        prism_top_z = base + pz
        self.tool_rest_z = prism_top_z + hz  # tool-centre height when resting on prisms

        # ---------------- tool (free body) --------------------------------
        self.tool = self._make_box(
            name="tool",
            half_size=(hx, hy, hz),
            rgba=[0.75, 0.75, 0.78, 1],
            tex_name="SteelScratched",
            fixed=False,
            density=self.tool_density,
        )
        # alias so inherited Lift code that references `self.cube` still works
        self.cube = self.tool

        # ---------------- fixed prisms + cubes ----------------------------
        self.prism_list = []
        self.cube_list = []
        for side, sign, tex in (("l", -1, "WoodRed"), ("r", +1, "WoodBlue")):
            x = ox + sign * x_prism

            prism = self._make_box(
                name=f"prism_{side}",
                half_size=(px / 2, py / 2, pz / 2),
                rgba=[0.45, 0.3, 0.2, 1],
                tex_name="WoodDark",
                fixed=True,
            )
            prism.get_obj().set("pos", f"{x} {oy} {base + pz / 2}")
            self.prism_list.append(prism)

            for ysign, ysuffix in ((-1, "neg"), (+1, "pos")):
                cube = self._make_box(
                    name=f"cube_{side}_{ysuffix}",
                    half_size=(c, c, c),
                    rgba=[1, 0, 0, 1] if side == "l" else [0, 0, 1, 1],
                    tex_name=tex,
                    fixed=True,
                )
                cube.get_obj().set("pos", f"{x} {oy + ysign * y_cube} {prism_top_z + c}")
                self.cube_list.append(cube)

        # Fixed bodies have no DOF, so their poses are set directly in the
        # MJCF above rather than through the placement sampler.

        # ---------------- placement sampler: ONLY for the tool ------------
        # reference z = prism top, so the tool's bottom starts z_offset above it
        ref_pos = np.array([ox, oy, prism_top_z])
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.tool)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ToolSampler",
                mujoco_objects=self.tool,
                x_range=[0.0, 0.0],  # centred between the two stands
                y_range=[0.0, 0.0],  # centred between the cube pairs
                rotation=0,  # aligned with x
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=ref_pos,
                z_offset=0.0005,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.tool] + self.prism_list + self.cube_list,
        )

    # ------------------------------------------------------------------
    # Sim references
    # ------------------------------------------------------------------
    def _setup_references(self):
        super(Lift, self)._setup_references()
        self.tool_body_id = self.sim.model.body_name2id(self.tool.root_body)

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------
    def _setup_observables(self):
        observables = super(Lift, self)._setup_observables()

        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"

        @sensor(modality=modality)
        def tool_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.tool_body_id])

        @sensor(modality=modality)
        def tool_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.tool_body_id], to="xyzw")

        @sensor(modality=modality)
        def gripper_to_tool_pos(obs_cache):
            if "tool_pos" in obs_cache and f"{pf}eef_pos" in obs_cache:
                return obs_cache[f"{pf}eef_pos"] - obs_cache["tool_pos"]
            return np.zeros(3)

        @sensor(modality=modality)
        def tool_lift(obs_cache):
            return np.array([self.sim.data.body_xpos[self.tool_body_id][2] - self.tool_rest_z])

        # order: [cube_l_neg, cube_l_pos, cube_r_neg, cube_r_pos]
        @sensor(modality=modality)
        def touching_cubes(obs_cache):
            return np.array([float(self.check_contact(self.tool, cube)) for cube in self.cube_list])

        # order: [prism_l, prism_r]  (support contacts)
        @sensor(modality=modality)
        def touching_prisms(obs_cache):
            return np.array([float(self.check_contact(self.tool, prism)) for prism in self.prism_list])

        sensor_fns = {
            "tool_pos": tool_pos,
            "tool_quat": tool_quat,
            "gripper_to_tool_pos": gripper_to_tool_pos,
            "tool_lift": tool_lift,
            "touching_cubes": touching_cubes,
            "touching_prisms": touching_prisms,
        }

        for name, fn in sensor_fns.items():
            observables[name] = Observable(name=name, sensor=fn, sampling_rate=self.control_freq)

        return observables

    # ------------------------------------------------------------------
    # Reward / success -- based only on the tool (the cubes and prisms are
    # fixed fixtures, not lift targets).
    # ------------------------------------------------------------------
    def reward(self, action=None):
        reward = 0.0

        tool_pos = np.array(self.sim.data.body_xpos[self.tool_body_id])

        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        dist = np.linalg.norm(gripper_site_pos - tool_pos)
        reward += 1 - np.tanh(10.0 * dist)

        # bonus once the tool is most of the way to the success height
        lift = tool_pos[2] - self.tool_rest_z
        if lift > 0.8 * self.lift_success_height:
            reward += 1.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0

        return reward

    def _check_success(self):
        tool_pos = self.sim.data.body_xpos[self.tool_body_id]
        return (tool_pos[2] - self.tool_rest_z) > self.lift_success_height
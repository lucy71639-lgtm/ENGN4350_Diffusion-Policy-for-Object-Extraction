"""

!! FORMAT CHANGED BUT RANDOMNESS CURRENTLY NOT IMPLEMENTED !!

LiftCubeBetweenFixedCubes: a robosuite environment based on Lift.

Setup: three cubes in a row.
    - cube_left  (red)   : FIXED to the world (no joint) -- immovable.
    - cube_free  (green) : the object to lift, has a free joint, starts
                            sandwiched between the two fixed cubes.
    - cube_right (blue)  : FIXED to the world (no joint) -- immovable.

The point of the task is that cube_free is in contact with both fixed
neighbors, so the robot has to deal with friction/contact resistance from
the fixed cubes while extracting/lifting it, rather than lifting an object
sitting in free space.

Key design decisions vs. a normal robosuite object:
  * cube_left / cube_right are created with `joints=[]`, which means
    robosuite attaches them rigidly to the worldbody with no degrees of
    freedom -- they behave like static fixtures (similar to how e.g. a
    cabinet or door frame is fixed in other robosuite envs).
  * Because they have no joint, they CANNOT be repositioned through
    `sim.data.set_joint_qpos` at reset time (which is how robosuite moves
    free objects). So they are NOT added to `placement_initializer`; instead
    their position is baked directly into their MJCF `pos` attribute in
    `_load_model`, computed from cube size + a `gap` parameter.
  * Only cube_free is registered with the placement sampler, and it's kept
    tightly bound to x=0 (centered exactly between the two fixed cubes),
    with just a little y / rotation jitter if you want domain
    randomization. Set `y_range`/`rotation` to 0 for a fully deterministic
    start.
  * `friction` is exposed as a constructor arg and applied to all three
    cube geoms. MuJoCo's default contact-friction combination rule takes
    the componentwise MAX of the two contacting geoms' friction arrays
    (sliding, torsional, rolling), so setting the same `friction` on all
    cubes directly controls cube-cube sliding friction. If you need
    asymmetric friction (e.g. free cube vs. left cube different from free
    cube vs. right cube) or an exact (non-max) combination rule, see the
    commented-out `<pair>` block at the bottom of `_load_model` -- MuJoCo
    lets you override the combined friction for a specific geom pair
    explicitly via a `<contact><pair .../></contact>` entry.

Tested against the robosuite 1.4.x / 1.5.x API shape. Attribute names like
`eef_site_id` / `sim.data.body_xpos` may differ slightly on other versions
-- check robosuite/environments/manipulation/lift.py in your installed
package if something doesn't resolve.
"""

import numpy as np
#from lxml import etree

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
import robosuite.utils.transform_utils as T

class LiftCube_RandGap(Lift):
    """
    Lift task variant: a free cube sandwiched between two fixed cubes.

    Args:
        cube_half_size (float): half-extent (m) of each cube (all three are
            the same size by default; edit `_load_model` if you want the
            fixed cubes to differ in size from the free one).
        gap (float): gap (m) between the free cube and each fixed neighbor
            at reset. 0.0 = exactly touching. Negative = slight interference
            / pre-loaded contact (mujoco will resolve the overlap on the
            first sim step -- keep this small, e.g. -0.001, if you want a
            "pressed together" start). Positive = a small gap, no contact
            until the robot pushes the free cube sideways.
        friction (3-tuple): (sliding, torsional, rolling) friction applied
            to all three cube geoms. Higher sliding friction -> harder for
            the free cube to slip past its fixed neighbors when lifted.
        y_range / rotation: passed to the free cube's placement sampler.
            Defaults keep it essentially fixed (centered, no rotation) --
            widen these if you want some randomization of the free cube's
            start pose within the gap.
        **kwargs: forwarded to Lift / ManipulationEnv (robots,
            controller_configs, has_renderer, etc.)
    """

    def __init__(
        self,
        cube_half_size=0.025,
        #gap=0.0,
        friction=(1.0, 0.005, 0.0001),
        y_range=(0.0, 0.0),
        rotation=0,
        **kwargs,
    ):
        self.cube_half_size = cube_half_size
        #self.gap = gap
        self.cube_friction = friction
        self.free_y_range = list(y_range)
        self.free_rotation = rotation
        super().__init__(**kwargs)

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

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}

        def make_material(tex_name):
            return CustomMaterial(
                texture=tex_name,
                tex_name=tex_name.lower(),
                mat_name=f"{tex_name.lower()}_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )

        s = self.cube_half_size

        # --- the liftable cube (free joint, placed via sampler) ---------
        self.cube_free = BoxObject(
            name="cube_free",
            size_min=[s, s, s],
            size_max=[s, s, s],
            rgba=[0, 1, 0, 1],
            material=make_material("WoodGreen"),
            friction=self.cube_friction,
        )
        # alias, in case any inherited Lift code references `self.cube`
        self.cube = self.cube_free

        # --- the two fixed cubes (no joint => rigid fixtures) -----------
        self.cube_left = BoxObject(
            name="cube_left",
            size_min=[s, s, s],
            size_max=[s, s, s],
            rgba=[1, 0, 0, 1],
            material=make_material("WoodRed"),
            friction=self.cube_friction,
            joints=[],  # no joint -> rigidly welded to worldbody
        )
        self.cube_right = BoxObject(
            name="cube_right",
            size_min=[s, s, s],
            size_max=[s, s, s],
            rgba=[0, 0, 1, 1],
            material=make_material("WoodBlue"),
            friction=self.cube_friction,
            joints=[],
        )

        # Fixed cubes have no DOF, so they can't be moved via
        # set_joint_qpos at reset time like the placement sampler normally
        # does. Instead we bake their position directly into the MJCF body
        # `pos` attribute right now, before the model is compiled.

        #gap = np.random.uniform(-0.005, 0.001)  # random gap between cubes
        #gap = -0.0005 #randomness de-implemented
        gap = 0.0
        #gap = -0.002
        #print('Gap:', gap)

        x_offset = 2 * s + gap  # center-to-center distance when touching + gap
        z = self.table_offset[2] + 0.001 + s  # table surface + small margin + half-height

        self.cube_left.get_obj().set("pos", f"{-x_offset} 0 {z}")
        self.cube_right.get_obj().set("pos", f"{x_offset} 0 {z}")

        # --- placement sampler: ONLY for the free cube -------------------
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube_free)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="FreeCubeSampler",
                mujoco_objects=self.cube_free,
                x_range=[0.0, 0.0],  # keep centered between the fixed cubes
                y_range=self.free_y_range,
                rotation=self.free_rotation,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube_free, self.cube_left, self.cube_right],
        )

        # ------------------------------------------------------------
        # OPTIONAL: explicit friction override for a specific geom pair.
        # MuJoCo's default rule combines two geoms' friction via
        # componentwise max, which is usually fine if all three cubes
        # share `self.cube_friction`. If you want the free-cube/left-cube
        # contact to have DIFFERENT friction than free-cube/right-cube (or
        # want an exact value regardless of max-combination rules),
        # uncomment and adapt this:
        #
        # contact_el = self.model.root.find("contact")
        # if contact_el is None:
        #     contact_el = etree.SubElement(self.model.root, "contact")
        # etree.SubElement(
        #     contact_el,
        #     "pair",
        #     geom1=self.cube_free.contact_geoms[0],
        #     geom2=self.cube_left.contact_geoms[0],
        #     friction="2.0 2.0 0.005 0.0001 0.0001",  # 5 numbers: slide(2) torsion roll
        #     condim="6",
        # )
        # ------------------------------------------------------------

    # ------------------------------------------------------------------
    # Sim references
    # ------------------------------------------------------------------
    def _setup_references(self):
        super(Lift, self)._setup_references()
        self.cube_free_body_id = self.sim.model.body_name2id(self.cube_free.root_body)
        self.cube_left_body_id = self.sim.model.body_name2id(self.cube_left.root_body)
        self.cube_right_body_id = self.sim.model.body_name2id(self.cube_right.root_body)

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------
    def _setup_observables(self):
        observables = super(Lift, self)._setup_observables()

        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"

        @sensor(modality=modality)
        def cube_free_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.cube_free_body_id])

        @sensor(modality=modality)
        def cube_free_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.cube_free_body_id], to="xyzw")

        @sensor(modality=modality)
        def gripper_to_cube_free_pos(obs_cache):
            if "cube_free_pos" in obs_cache and f"{pf}eef_pos" in obs_cache:
                return obs_cache[f"{pf}eef_pos"] - obs_cache["cube_free_pos"]
            return np.zeros(3)

        @sensor(modality=modality)
        def touching_left(obs_cache):
            return np.array(
                [float(self.check_contact(self.cube_free, self.cube_left))]
            )

        @sensor(modality=modality)
        def touching_right(obs_cache):
            return np.array(
                [float(self.check_contact(self.cube_free, self.cube_right))]
            )

        sensors = [
            cube_free_pos,
            cube_free_quat,
            gripper_to_cube_free_pos,
            touching_left,
            touching_right,
        ]
        names = [
            "cube_free_pos",
            "cube_free_quat",
            "gripper_to_cube_free_pos",
            "touching_left",
            "touching_right",
        ]

        for name, s in zip(names, sensors):
            observables[name] = Observable(name=name, sensor=s, sampling_rate=self.control_freq)

        return observables

    # ------------------------------------------------------------------
    # Reward / success -- based only on cube_free (the fixed cubes aren't
    # lift targets).
    # ------------------------------------------------------------------
    def reward(self, action=None):
        reward = 0.0

        table_height = self.table_offset[2]
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_free_body_id])

        #print(self.robots[0].eef_site_id["right"])
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        dist = np.linalg.norm(gripper_site_pos - cube_pos)
        reward += 1 - np.tanh(10.0 * dist)

        height_above_table = cube_pos[2] - table_height
        if height_above_table > 0.08:
            reward += 1.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0

        return reward

    def _check_success(self):
        table_height = self.table_offset[2]
        cube_pos = self.sim.data.body_xpos[self.cube_free_body_id]
        return (cube_pos[2] - table_height) > 0.1



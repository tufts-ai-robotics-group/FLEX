from robosuite.utils.binding_utils import MjRenderContext, MjRenderContextOffscreen
import mujoco
from typing import Callable

class MjRendererForceVisualization(MjRenderContext):
    '''
    A new rendering class that can modify scene 

    init: need a modify_fn(scene), it takes a scene and modify it
    '''
    def __init__(self, sim, modify_fn:Callable, device_id, max_width=640, max_height=480):
        super().__init__(sim, offscreen=True, device_id=device_id, max_width=max_width, max_height=max_height)
        self.modify_fn = modify_fn

    def render(self, width, height, camera_id=None, segmentation=False):
        viewport = mujoco.MjrRect(0, 0, width, height)

        # if self.sim.render_callback is not None:
        #     self.sim.render_callback(self.sim, self)

        # update width and height of rendering context if necessary
        if width > self.con.offWidth or height > self.con.offHeight:
            new_width = max(width, self.model.vis.global_.offwidth)
            new_height = max(height, self.model.vis.global_.offheight)
            self.update_offscreen_size(new_width, new_height)

        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(
            self.model._model, self.data._data, self.vopt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn
        )

        self.modify_fn(self.scn)

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

        # for marker_params in self._markers:
        #     self._add_marker_to_scene(marker_params)

        mujoco.mjr_render(viewport=viewport, scn=self.scn, con=self.con)
        # for gridpos, (text1, text2) in self._overlay.items():
        #     mjr_overlay(const.FONTSCALE_150, gridpos, rect, text1.encode(), text2.encode(), &self._con)

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0


class MjRendererForceVisualizationOffscreen(MjRenderContext):
    '''
    A new rendering class that can modify scene 

    init: need a modify_fn(scene), it takes a scene and modify it
    '''
    def __init__(self, sim, modify_fn:Callable, device_id, max_width=640, max_height=480):
        super().__init__(sim, offscreen=True, device_id=device_id, max_width=max_width, max_height=max_height)
        self.modify_fn = modify_fn

    def render(self, width, height, camera_id=None, segmentation=False):
        viewport = mujoco.MjrRect(0, 0, width, height)

        # if self.sim.render_callback is not None:
        #     self.sim.render_callback(self.sim, self)

        # update width and height of rendering context if necessary
        if width > self.con.offWidth or height > self.con.offHeight:
            new_width = max(width, self.model.vis.global_.offwidth)
            new_height = max(height, self.model.vis.global_.offheight)
            self.update_offscreen_size(new_width, new_height)

        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(
            self.model._model, self.data._data, self.vopt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn
        )

        self.modify_fn(self.scn)

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

        # for marker_params in self._markers:
        #     self._add_marker_to_scene(marker_params)

        mujoco.mjr_render(viewport=viewport, scn=self.scn, con=self.con)
        # for gridpos, (text1, text2) in self._overlay.items():
        #     mjr_overlay(const.FONTSCALE_150, gridpos, rect, text1.encode(), text2.encode(), &self._con)

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0
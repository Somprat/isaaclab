# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the exact motion-generator URDF into a separate IsaacLab USD asset."""

import hashlib
import importlib.metadata
import json
import xml.etree.ElementTree as ET

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

from isaaclab_tasks.core.velocity.config.g1.mpc_model import MPC_ASSET_DIR, MPC_URDF, read_urdf_model


def author_contact_material(path):
    """Make collision material explicit, including originally instanced meshes."""
    from pxr import Usd, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(path))
    while True:
        instances = [prim for prim in stage.Traverse() if prim.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    material = UsdShade.Material.Define(stage, str(stage.GetDefaultPrim().GetPath()) + "/MpcContactMaterial")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(0.4)
    physics.CreateDynamicFrictionAttr(0.4)
    physics.CreateRestitutionAttr(0.0)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    stage.GetRootLayer().Save()


def main():
    # An installed Kit distribution shadows the isolated importer's namespace.
    # Prefer the standalone package for this offline conversion only.
    import isaacsim

    isolated = importlib.metadata.distribution("isaacsim-asset-isolated")
    isaacsim.__path__.insert(0, str(isolated.locate_file("isaacsim")))
    model = read_urdf_model()
    cfg = UrdfConverterCfg(
        asset_path=str(MPC_URDF),
        usd_dir=str(MPC_ASSET_DIR),
        fix_base=False,
        merge_fixed_joints=True,
        force_usd_conversion=True,
        ros_package_paths=[{"name": "g1_description", "path": str(MPC_URDF.parent.parent)}],
        collision_from_visuals=False,
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    )
    asset = UrdfConverter(cfg)
    author_contact_material(asset.usd_path)
    model["generated_usd"] = asset.usd_path
    model["asset_files_sha256"] = {
        str(path.relative_to(MPC_ASSET_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in MPC_ASSET_DIR.rglob("*.usd*")
        if path.is_file()
    }
    meshes = {
        mesh.get("filename").replace("package://g1_description/", "")
        for mesh in ET.parse(MPC_URDF).getroot().iter("mesh")
    }
    model["source_meshes_sha256"] = {
        relative: hashlib.sha256((MPC_URDF.parent.parent / relative).read_bytes()).hexdigest() for relative in meshes
    }
    (MPC_ASSET_DIR / "model_manifest.json").write_text(json.dumps(model, indent=2) + "\n")
    print(
        json.dumps(
            {
                "usd": asset.usd_path,
                "mass_kg": model["total_mass"],
                "rigid_bodies": len(model["bodies"]),
                "joints": len(model["joints"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
source /opt/ros/${ROS_DISTRO:-humble}/setup.bash
export CMAKE_PREFIX_PATH="/opt/pinocchio-2.7:${CMAKE_PREFIX_PATH:-}"
export PKG_CONFIG_PATH="/opt/pinocchio-2.7/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="/opt/pinocchio-2.7/lib:${LD_LIBRARY_PATH:-}"

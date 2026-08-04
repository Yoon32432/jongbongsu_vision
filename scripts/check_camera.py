from bagvision.realsense_capture import RealSenseCamera


def main():
    camera = RealSenseCamera()
    camera.start()
    try:
        color, depth, intr = camera.get_frames()
        print(f"color shape: {color.shape}, depth shape: {depth.shape}")
        print(f"intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} ppx={intr.ppx:.1f} ppy={intr.ppy:.1f} depth_scale={intr.depth_scale}")
        cy, cx = depth.shape[0] // 2, depth.shape[1] // 2
        center_depth_m = depth[cy, cx] * intr.depth_scale
        print(f"center pixel depth: {center_depth_m:.3f} m")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()

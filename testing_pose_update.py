def update_position(self, current_frame, previous_frame):

    kp1, des1 = orb.detectAndCompute(previous_frame, None)
    kp2, des2 = orb.detectAndCompute(current_frame, None)


    # Create a BFMatcher object and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography or essential matrix to estimate movement
    # Note: This step might be replaced with more sophisticated methods
    # depending on the accuracy required and the nature of the movement.
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Update the current position
    # This is a simplified approach. In real applications, you would need
    # to convert this transformation into a real-world metric, which can be complex.
    if H is not None:
        # Decompose the homography matrix to extract translation
        _, _, trans, _, _, _, _ = cv2.decomposeHomographyMat(H)
        # Example: Update position assuming translation is the last column
        # This is a simplification and may not be accurate.
        current_position += trans[-1].reshape(3)

    return current_position
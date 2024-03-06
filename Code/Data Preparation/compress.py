import cv2
import ffmpeg
import face_recognition

def detect_and_resize_face(frame, target_size=(256, 256)):
    # Use face_recognition to locate faces in the frame
    face_locations = face_recognition.face_locations(frame)

    if face_locations:
        # Assuming only one face, get the first face location
        top, right, bottom, left = face_locations[0]
        
        # Crop the face region from the frame
        face = frame[top:bottom, left:right]

        # Resize the face to the target size
        resized_face = cv2.resize(face, target_size)

        return resized_face

    # If no face is detected, return the original frame
    return frame

def compress_video(input_file, output_file, bitrate, codec='libx265'):
    input_stream = ffmpeg.input(input_file)
    output_stream = ffmpeg.output(input_stream.video.filter('scale=640:-1', filter_name='scale'), output_file,
                                  vf='fps=30', video_bitrate=f'{bitrate}k', vcodec=codec)
    ffmpeg.run(output_stream)

def main():
    input_file = 'videos/obama001.mp4'

    bitrates = [60, 70, 80, 90, 100, 110, 120]

    for bitrate in bitrates:
        output_file = f'compressed/output_{bitrate}kbps_h265_faces.mp4'

        # Open the video file
        video_capture = cv2.VideoCapture(input_file)
        
        # Get the frames per second (fps) of the input video
        fps = video_capture.get(cv2.CAP_PROP_FPS)

        # Create a VideoWriter object to write the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (256, 256))

        # Process each frame
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Detect and resize face
            resized_face = detect_and_resize_face(frame)

            # Write the processed frame to the output video
            out.write(resized_face)

        # Release video capture and writer
        video_capture.release()
        out.release()

        print(f'Compression at {bitrate} kbps (H.265) with face resizing complete. Output: {output_file}')

if __name__ == "__main__":
    main()

# OpenCV Project 🧾
- Final commit: `809a9bd9de472fccb14988920784b4e6711ca35f`
- Project located within `project` folder
- Run Python script labeled `main.py`
- Images are in the `images` folder but feel free to use others, just make sure to put them within that folder
  - Use receipt images like `receipt.png` and `receipt1.jpg` for the my document scanner and text detector
  - Use qr code images like `qr.png` for my qr scanner
## Independence + Curiosity 💖
I did not use histograms (chapter 7) in my project, so I went beyond in a few aspects. 
- Added 3 extra components
  - `documentScanner()`
    - Used grayscale, edge detection, contours, perspective transform, and adaptive thresholding to scan a paper
    - Can incoporate web cam if desired, however I did not have one so I didn't do that
  - `textDetector()`
    - Used Pytesseract's library in order to outline text + return it to the command prompt
  - `qrScanner()`
    - Used OpenCV's built in `QRCodeDetector()` to return link to QR code uploaded
## Chapters Included 📌
I commented where each is used in my code with the 📌 emoji for reference.
- 4.3: Accessing and Manipulating Pixels
- 5.1: Lines and Rectangles 
- 6.1.3: Resizing
- 6.3: Bitwise Operations
- 6.6: Color Spaces
- 8.2: Gaussian Blurring
- 8.3: Median Blurring
- 9.2: Adaptive Thresholding
- 10.2: Canny Edge Detection
- 11.1: Counting Coins
- 11.2: Contours and OpenCV Version Caveats

# Sudoku_solver

Solving Sudoku with OpenCV_Python_CNN  

Sudoku_raw_image:  
<img src="https://github.com/Parimala6/Sudoku_solver/blob/main/images/sudoku4.jpg" width="300">

Converting image to gray scale to reduce information:  
<img src="https://github.com/Parimala6/Sudoku_solver/blob/main/images/Gray_scale.jpg" width="300">

Blur the image to remove noise:  
(Smoothing the image with Gaussian filter, a low-pass filter that reduces the high-frequency components)
<img src="https://github.com/Parimala6/Sudoku_solver/blob/main/images/Blur.jpg" width="300">

Thresholding the image to seperate foreground pixels from background pixels (minimizing the background noise):  
Adaptive thresholding - the threshold value is calculated for smaller regions and therefore, there will be different threshold values for different regions.  
<img src="https://github.com/Parimala6/Sudoku_solver/blob/main/images/Threshold.jpg" width="300">

Inverting colors so as to find the contours properly:  
<img src="https://github.com/Parimala6/Sudoku_solver/blob/main/images/Inverted_color.jpg" width="300">

Finding the largest contour in the image i.e., sudoku:  
<img src="https://github.com/Parimala6/Sudoku_solver/blob/main/images/Largest_contour.jpg" width="300">

Finding the corners and extracting the ROI:  
<img src="https://github.com/Parimala6/Sudoku_solver/blob/main/images/Extracted_sudoku.jpg" width="300">

Extracting all the cells of the grid:
<img src="https://github.com/Parimala6/Sudoku_solver/blob/main/images/Grid.jpg" width="300">

Extracted cell:  
<img src="https://github.com/Parimala6/Sudoku_solver/blob/main/gridcells/cell03.jpg" width="300">

Extracting and recognizing the digits in the cell:  
<img src="https://github.com/Parimala6/Sudoku_solver/blob/main/Cleanedcells/cell03.png" width="300">

Solved Sudoku:  
<img src="https://github.com/Parimala6/Sudoku_solver/blob/main/images/Solved_sudoku.jpg" width="300">  

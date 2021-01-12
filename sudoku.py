import cv2
import numpy as np
from tensorflow.python.keras.models import load_model
import constraint

##Function to read the image
def read_img():
##    print('Please show the puzzel to the camera')
    image_url = 'images/sudoku4.jpg'
    img = cv2.imread(image_url)
    return img

##Scales and centres an image onto a new background square
def scale_and_centre(img, size, margin=30, background=0):
    h, w = img.shape[:2]
    ##Handles centering for a given length that may be odd or even
    def centre_pad(length):
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = ratio*w, ratio*h
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = ratio*w, ratio*h
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None)
    return cv2.resize(img, (size, size))

    
##Recognize the digits in the sudoku
def predict(img_grid):
    img = img_grid.copy()
    img = cv2.resize(img, (28,28))
    img = img.astype('float32')
    img = img.reshape(1,28,28,1)
    img /= 255.0

    model = load_model('model.h5')
    pred = model.predict(img, batch_size=1)

    return pred.argmax()

##Pre processing the image
#Convert image to gray scale
img_n = read_img()
img = cv2.cvtColor(img_n, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', img)
##cv2.imwrite('images/Gray_scale.jpg', img)

#Blur the image using Gaussian blur to reduce noise
blur_img = cv2.GaussianBlur(img.copy(), (9,9), 0)
cv2.imshow('blur', blur_img)
##cv2.imwrite('images/Blur.jpg', blur_img)

#Segmentation - thresholding
thr_img = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
##thr_img = cv2.threshold(blur_img, 255, 255, cv2.THRESH_BINARY)[1]
cv2.imshow('threshold', thr_img)
##cv2.imwrite('images/Threshold.jpg', thr_img)

#Invert colors to extract the grid
invt_img = cv2.bitwise_not(thr_img, thr_img)
cv2.imshow('invert', invt_img)
##cv2.imwrite('images/Inverted_color.jpg', invt_img)

#Use of Gaussian threshold reduces noise but shrinks the image. So, dilate it.
##kernel = np.ones((3,3), np.uint8)
##kernel = np.array([[0.,1.,0.], [1.,1.,1.], [0.,1.,0.]], np.uint8)
##dil_img = cv2.dilate(invt_img, kernel)
##cv2.imshow('dilate', dil_img)

##Find and read the puzzle
#Find the largest external contour
ext_contour,_ = cv2.findContours(invt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##ext_contour = ext_contour[0] if len(ext_contour) == 2 else ext_contour[1]
ext_contour = sorted(ext_contour, key=cv2.contourArea, reverse=True)[:1]
cont_img = cv2.drawContours(img_n, ext_contour, -1, (0,255,0), 5)
#-1 signifies drawing all contours
cv2.imshow('contours', cont_img)
##cv2.imwrite('images/Largest_contour.jpg', cont_img)
##print(len(ext_contour))

#Find corners
for c in ext_contour:
##    x, y, w, h = cv2.boundingRect(c)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    roi = cv2.drawContours(img_n, [box], 0, (0,0,255), 2)
    cv2.imshow('ROI', roi)
    wd = int(rect[1][0])
    ht = int(rect[1][1])
    print(rect[2])

    if rect[2] in [90.0, -90.0]:
        
        dimensions = np.array([[0,0], [wd-1,0], [wd-1,ht-1], [0,ht-1]], np.float32)
        ordered_corners = box.astype('float32')

        grid = cv2.getPerspectiveTransform(ordered_corners, dimensions)
        ext_img = cv2.warpPerspective(img_n, grid, (wd,ht))

    else:
        print('Please rotate the image accordingly')
        break
        
cv2.imshow('extracted sudoku', ext_img)
##cv2.imwrite('images/Extracted_sudoku.jpg', ext_img)


##Extracting cells
grid = np.copy(ext_img)
grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
grid = cv2.bitwise_not(grid, grid)
cv2.imshow('grid', grid)
##cv2.imwrite('images/Grid.jpg', grid)

grid_h = np.shape(grid)[0]
grid_w = np.shape(grid)[1]
##print(grid_h, grid_w)
cell_h = grid_h // 9
cell_w = grid_w // 9
##print(cell_h, cell_w)

tempgrid = []
for i in range(cell_h, grid_h+1, cell_h):
    for j in range(cell_w, grid_w+1, cell_w):
        rows = grid[i-cell_h : i]
        tempgrid.append([rows[k][j-cell_w : j] for k in range(len(rows))])

finalgrid = []
for i in range(0, len(tempgrid)-8, 9):
    finalgrid.append(tempgrid[i : i+9])

for i in range(9):
    for j in range(9):
        finalgrid[i][j] = np.array(finalgrid[i][j])

##Remove if any previous images
try:
    for i in range(9):
        for j in range(9):
            os.remove('gridcells/cell' + str(i) + str(j) + '.jpg')
except:
    pass

##Save the cell images
for i in range(9):
    for j in range(9):
        cv2.imwrite(str('gridcells/cell' + str(i) + str(j) + '.jpg'), finalgrid[i][j])
##print('Saved the cells')

##Extracting the digits
tmp_grid = [[0 for i in range(9)] for j in range(9)]
for i in range(9):
    for j in range(9):
        img = finalgrid[i][j]
        img = cv2.resize(img, (28, 28))
##        cv2.imshow('final_grid_cells', img)
##        cv2.waitKey(1500)
        
        thr = 128 #greyscale
        img_gray = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)[1]

        conts = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        conts = conts[0] if len(conts) == 2 else conts[1]

        for c in conts:
            x,y,w,h = cv2.boundingRect(c)
            if (x<3 or y<3 or h<3 or w<3):
                continue
            ROI = img_gray[y:y+h, x:x+w]
            ROI = scale_and_centre(ROI, 90)

            cv2.imwrite('Cleanedcells/cell{}{}.png'.format(i,j), ROI)
##            cv2.imshow('cells{}{}'.format(i,j), ROI)
            tmp_grid[i][j] = predict(ROI)
##            print(tmp_grid[i][j])
##            cv2.waitKey(3000)
##print(tmp_grid)
            

##Solving the sudoku
problem = constraint.Problem()

#Letting variables 11 to 99 have an interval of [1..9]
for i in range(1, 10):
    problem.addVariables(range(i*10+1, i*10+10), range(1, 10))

#All values in a row must be different
#11 through 19 must be different, 21 through 29 must be all different
for i in range(1, 10):
    problem.addConstraint(constraint.AllDifferentConstraint(), range(i*10+1, i*10+10))

#All values in a column must be different
#11,21,31...91 must be different, also 12,22,32...92 must be different
for i in range(1, 10):
    problem.addConstraint(constraint.AllDifferentConstraint(), range(10+i, 100+i, 10))

#The nine 3x3 squares must have all different values,
#Note that each square starts at row indices 1, 4, 7
for i in [1,4,7]:
    #Same for columns, the squares start at indices 1, 4, 7 
    #one square starts at 11, the other at 14, another at 41, etc
    for j in [1,4,7]:
        square = [10*i+j,10*i+j+1,10*i+j+2,10*(i+1)+j,10*(i+1)+j+1,10*(i+1)+j+2,
                  10*(i+2)+j,10*(i+2)+j+1,10*(i+2)+j+2]
        #As an example, for i = 1 and j = 1 (bottom left square), the cells 11,12,13,
        #21,22,23, 31,32,33 have to be all different
        problem.addConstraint(constraint.AllDifferentConstraint(), square)

#Adding a constraint for each number already present on the board (0 is an empty cell)
for i in range(9):
    for j in range(9):
        if tmp_grid[i][j] != 0:
            def c(variable_value, value_in_table = tmp_grid[i][j]):
                if variable_value == value_in_table:
                    return True
            problem.addConstraint(c, [((i+1)*10 + (j+1))])

sol = problem.getSolutions()
solved_sudoku = [[0 for i in range(9)] for j in range(9)]

if len(sol) == 0:
    print('No solutioons found :(')
else:
    sol = sol[0]
    for i in range(1, 10):
        for j in range(1, 10):
            solved_sudoku[i-1][j-1] = (sol[i*10+j])
    print('Sudoku solved!')
##    print(solved_sudoku)

    #print the solution on the grid
    for i in range(9):
        for j in range(9):
            if tmp_grid[i][j] == 0:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (255, 0, 255)
                thickness = 2
                cv2.putText(grid, str(solved_sudoku[i][j]), (int((j+0.5)*cell_h), int((i+0.8)*cell_w)),
                            font, font_scale, color, thickness)

    cv2.imshow('solved sudoku', grid)
    ##cv2.imwrite('images/Solved_sudoku.jpg', grid)




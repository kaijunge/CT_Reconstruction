import numpy as np 
import matplotlib.pyplot as plt
import cv2
import math
from scipy.fftpack import fft, fftshift, ifft

# display the image 
def dispimg(img):
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  
    plt.show(block=False)
    print("Press enter")
    input()
    plt.close()


# scale the image to have the largest dimension equal to 2000 pixels
def rescale(img, longest_dimension = 2000):
    max_dimension = max(img.shape[0], img.shape[1])
    scale = longest_dimension/max_dimension
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    img_dim = (width, height)
    # resize image
    img = cv2.resize(img, img_dim, interpolation = cv2.INTER_AREA)
    
    return img

# create ramlak filter (in the frequency domain)
def ram_lak_filter(n):
    filt = np.zeros(n)
    
    inc = (2*np.pi)/(n-n%2)
    offset  = 0
    for i in range(n):
        if n%2 ==0 and i == n/2:
            offset = 1
        filt[i] = abs(-1*np.pi + inc *(i+offset))
        
    return filt
    
# return correctly formatted filter in the spacial and freqency domain 
# filter_n -> number of elements in the frequency domain 
# conv_n -> 1/2 of the number of elements in the FIR filter used in convolution 
def produce_filter(filter_n, half_conv_n):
    ramlak_correctly_scaled = fftshift(ram_lak_filter(filter_n))
    IFT_ramlak = np.real(np.around(np.fft.ifft(ramlak_correctly_scaled), decimals = 3))
    q = np.concatenate([IFT_ramlak[half_conv_n:0:-1], IFT_ramlak[0:half_conv_n+1]])
    
    return ramlak_correctly_scaled, q


def scan_image(img, angle_division, q):

    print("Starting scanning process\n")
    # extract number of rows and colums from the scaled image (longest dimension = 2000 pixels)
    num_rows, num_cols = img.shape[:2]


    N = int(180/angle_division)
    ten_percent = math.ceil(N/10)
    sinogram_store = []
    conv_store = []

    scale = 1.5

    for n in range(N):
        angle = n * angle_division

        # translation and rotation matrix
        translation_matrix = np.float32([ [1,0,int((scale-1)*0.5*num_cols)], [0,1,int((scale-1)*0.5*num_rows)] ])
        rotation_matrix = cv2.getRotationMatrix2D((int(scale*0.5*num_cols), int(scale*0.5*num_rows)), angle, 1) 

        # perform translate and rotate on a white background
        img_translation = cv2.warpAffine(img, translation_matrix, (int(scale*num_cols), int(scale*num_rows)), borderValue=(255,255,255))
        im = cv2.warpAffine(img_translation, rotation_matrix, (int(scale*num_cols), int(scale*num_rows)), borderValue=(255,255,255)) 

        # find the "integral" along the vertical direction
        if n == 0:
            sinogram = np.matmul(np.full(im.shape[0], -1), np.subtract(im, np.full(im.shape, 255)))
            sinogram_store.append(sinogram)
            conv_store.append(np.convolve(q, sinogram))
        else:
            sinogram_current = np.matmul(np.full(im.shape[0], -1), np.subtract(im, np.full(im.shape, 255)))
            sinogram = np.vstack((sinogram_current, sinogram))

            sinogram_store.append(sinogram_current)
            conv_store.append(np.convolve(q, sinogram_current))

        #print(str(n) + "deg ", end= "")
        if n > 0 and n%ten_percent == 0:
            print(str(math.floor(n/ten_percent)*10) + " percent done")
           
    print("100 percent done\n")      
    print("Producing sinogram\n\n")
    sinogram = sinogram.astype(float)
    
    height, width = sinogram.shape
    height =  height *width / N
    dim = (int(width), int(height))
    
    # resize image for visual purposes
    newimg = cv2.resize(sinogram, dim, interpolation = cv2.INTER_AREA)
    
    return conv_store, sinogram_store, newimg


def complete_filtering(mode, conv_store, sino_store, half_conv_n):
    
    print("Producing two convolution results conv(projection(phi), IFT(ram-lak))")
    print("mode 1: by direct convolution")
    print("mode 2: DFT -> Multiply -> iDFT\n")
    #Filtering using DFT -> multipy -> iDFT
    conv_store_2 = []
    FT_q = fftshift(ram_lak_filter(len(sino_store[0])))
    for i in range(len(sino_store)):

        p = sino_store[i]
        FT_p = fft(p)
        conv_store_2.append(np.real(ifft(FT_p * FT_q)))

    #Get rid of convolution zeros in conv_store
    conv_store_1 = []
    for i in range(len(conv_store)):
        conv_store_1.append(conv_store[i][half_conv_n:len(conv_store[i])-half_conv_n])
        
    if mode == 1:
        print("mode 1 was chosen")
        return conv_store_1
    else:
        print("mode 2 was chosen")
        return conv_store_2


def backproject(conv, angle_division, scaled_img):
    
    print("\nBegin backprojection")

    angle_division = int(180/len(conv))
    projection = 0
    N = len(conv)
    five_percent = math.ceil(N/20)
    
    for i in range(N):

        angle = i * angle_division
        conv_i = conv[i]
        projected_conv = np.array([conv_i,]*len(conv_i))

        num_rows, num_cols = projected_conv.shape[:2]

        scale = 1.5

        # translation and rotation matrix
        translation_matrix = np.float32([ [1,0,int((scale-1)*0.5*num_cols)], [0,1,int((scale-1)*0.5*num_rows)] ])
        rotation_matrix = cv2.getRotationMatrix2D((int(scale*0.5*num_cols), int(scale*0.5*num_rows)), -1*angle, 1) 

        # perform translate and rotate
        img_translation = cv2.warpAffine(projected_conv, translation_matrix, (int(scale*num_cols), int(scale*num_rows)))#, borderValue=(0,0,0))
        im = cv2.warpAffine(img_translation, rotation_matrix, (int(scale*num_cols), int(scale*num_rows)), borderValue=(0,0,0)) 

        if i == 0:
            projection = np.zeros((im.shape)).astype(float)
        projection += im.astype(float)
        
        if i > 0 and i%five_percent == 0:
            print(str(int(i/five_percent)*5) + " percent done")
           
    print("100 percent done\n")
    d = len(projection)
    half_height = scaled_img.shape[0]/2
    half_width = scaled_img.shape[1]/2
    projection = np.round((projection-np.amin(projection))/np.ptp(projection)*255)
    projection = projection[int(d/2-half_height):int(d/2+half_height)
                            ,int(d/2-half_width):int(d/2+half_width)]
    
    correct_projection = 255 - projection
    
    return correct_projection

# save to current directory
def saveimg(img, name):
    cv2.imwrite(name + ".png", img)


def dispresults(img1, img2, img3):

    print("Displaying key results")
    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(img1, cmap = 'gray', interpolation = 'bicubic')
    ax[1].imshow(img2, cmap = 'gray', interpolation = 'bicubic')
    ax[2].imshow(img3, cmap = 'gray', interpolation = 'bicubic')

    plt.setp(ax, xticks=[], yticks=[])
   
    ax[0].title.set_text('Original Image')
    ax[1].title.set_text('Reconstructed Image')
    ax[2].title.set_text('Sinogram')
    plt.show()


if __name__ == "__main__":

    # Tunable parameters
    largest_img_dim = 2000     # Pixel size of max dimension in the image
    FIR_length = 100           # Number of elements in the FIR filter for convolution
    angle = 1                  # Angle increment for scanning

    # Load image
    img = cv2.imread('arrow.png',0).astype(float)
    #dispimg(img)

    # Rescale image to desired dimension
    scaled_img = rescale(img, longest_dimension = largest_img_dim)

    # determine FIR filter and Ram-Lak filter (in freq domain)
    mod_omega, q = produce_filter(int(largest_img_dim*1.5), int(FIR_length/2))

    # Scan the image and obtain the sinogram. Also perform direct convolution with for each projection at every angle of scan
    conv_store, sino_store, sinogram = scan_image(scaled_img, angle, q)
    #dispimg(sinogram)

    # Complete the filtering proecess
    conv = complete_filtering(1, conv_store, sino_store, int(FIR_length/2))

    # Back project the convolved sinogram 
    reconstruction = backproject(conv, angle, scaled_img)
    #dispimg(reconstruction)

    # Results
    saveimg(reconstruction, "reconstruction")
    dispresults(img, reconstruction, sinogram)
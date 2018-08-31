# coding: utf-8
# Autor: Ricardo Antonello 
# Site: cv.antonello.com.br
# E-mail: ricardo@antonello.com.br

# import the necessary packages
import time
import cv2
print(cv2.__version__)
import numpy as np
from matplotlib import pyplot as plt

#import os
#import sys
#print(sys.executable)

threshold_GLOBAL = 160

def texto(img, texto, coord, fonte = cv2.FONT_HERSHEY_SIMPLEX, cor=(0,0,255), tamanho=0.7, thickness=2):
    textSize, baseline = cv2.getTextSize(texto, fonte, tamanho, thickness);
    cor_background = 0
    if type(cor)==int: # se não for colorida a imagem
        cor_background=255-cor
    else:
        cor_background=(255-cor[0],255-cor[1],255-cor[2])
    #print(cor_background)
    cv2.rectangle(img, (coord[0], coord[1]-textSize[1]-3), (coord[0]+textSize[0], coord[1]+textSize[1]-baseline), cor_background, -1)
    #cv2.putText(img, texto, coord, fonte, tamanho, cor_background, thickness+1, cv2.LINE_AA)
    cv2.putText(img, texto, coord, fonte, tamanho, cor, thickness, cv2.LINE_AA)
    return img

def exibe(i,j=None,t1='Imagem',t2='Imagem Modificada'):
    plt.figure(figsize=(16, 50)) # LARGGura e ALTura da imagem total em polegadas #fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    plt.subplot2grid((1,2),(0,0)) #(linhas, colunas) do grid (l,c) do elemento
    plt.title(t1)
    plt.imshow(i)
    plt.subplot2grid((1,2),(0,1)) #(linhas, colunas) do grid (l,c) do elemento
    plt.title(t2)
    plt.imshow(j)

def filtros(img):
    blur = cv2.medianBlur(img, 15) #img_blobs = cv2.GaussianBlur(img_blobs, (7,7), 0) # aplica blur     #img_blobs = cv2.bilateralFilter(img, 3, 21, 21)
    #exibe(img, blur, t1="Imagem original", t2="Blur")
    #pb = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY) # converte para PB
    pb = blur.copy()
    #exibe(img, pb)
    # Threshold ou Binarização
    (T, bin) = cv2.threshold(pb, threshold_GLOBAL, 255, cv2.THRESH_BINARY)
    #exibe(img, bin, t1="Imagem original", t2="Threshold")
    ### Inverte preto e branco
    img_inv = 255-bin # inverte a imagem
    #exibe(img, img_inv, t1="Imagem original", t2="Imagem invertida")
    return img_inv, blur

def detecta(img, _minThreshold=120, _maxThreshold=255, _minArea=3000, _maxArea=300000, _minCircularity=0.9, _maxCircularity=1.0, 
                 _minConvexity=0.5, _maxConvexity=1.0, _minInertiaRatio=0.9, _maxInertiaRatio=1.0):
    
    params = cv2.SimpleBlobDetector_Params() # Setup SimpleBlobDetector parameters.
    params.minThreshold = _minThreshold; # Change thresholds
    params.maxThreshold = _maxThreshold;
    
    params.filterByArea = True # Filter by Area.
    params.minArea = _minArea
    params.maxArea = _maxArea
    
    params.filterByCircularity = True # Filter by Circularity # This means that a circle has a circularity of 1, circularity of a square is 0.785, and so on.
    params.minCircularity = _minCircularity
    params.maxCircularity = _maxCircularity
    
    params.filterByConvexity = True # Filter by Convexity
    params.minConvexity = _minConvexity
    params.maxConvexity = _maxConvexity
    
    params.filterByInertia = True # Filter by Inertia #for a circle, this value is 1, for an ellipse it is between 0 and 1, and for a line it is 0. 
    params.minInertiaRatio = _minInertiaRatio
    params.maxInertiaRatio = _maxInertiaRatio

    detector = cv2.SimpleBlobDetector_create(params)
    inicio = time.time()
    # Detecta blobs
    keypoints = detector.detect(img) 
    #print(keypoints)
    #print("Detectando Blobs... Tempo: %.2f segundos" % (time.time()-inicio))
    img_blobs=cv2.drawKeypoints(img, keypoints, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DEFAULT) #For Python API, flags are modified as cv2.DRAW_MATCHES_FLAGS_DEFAULT, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG, cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    #for k in keypoints:
    #   cv2.circle(img_blobs, (int(k.pt[0]), int(k.pt[1])), int(k.size)//2, 255, 3);
    return img_blobs, keypoints

def encontraCor(img): #imagem deve estar em formato RGB
    red_pixels = 0
    for p in np.ravel(img[::4,::4,0]):
        if p>200:
            red_pixels+=1
    green_pixels = 0
    for p in np.ravel(img[::4,::4,1]):
        if p>200:
            green_pixels+=1
    blue_pixels = 0
    for p in np.ravel(img[::4,::4,2]):
        if p>200:
            blue_pixels+=1
    CORTE = 150
    #print('r',red_pixels,'g',green_pixels,'b',blue_pixels)
    if red_pixels>CORTE and green_pixels>CORTE:
        return "Amarelo"
    elif red_pixels>CORTE:
        return "Vermelho"
    elif blue_pixels>CORTE:
        return "Azul"    
    else:
        return ''

def isQuadrado(img):
    img_blobs, keypoints = detecta(img, _minCircularity=0.7, _maxCircularity=0.85, _minConvexity=0.8, _maxConvexity=1.1, 
                        _minInertiaRatio=0.9, _maxInertiaRatio=1.0)
    #exibe(img, img_blobs, t2='Quadrado')
    return (True, img_blobs) if len(keypoints)>0 else (False, img_blobs)

def isCirculo(img):
    img_blobs, keypoints = detecta(img, _minCircularity=0.85, _maxCircularity=1.2, _minConvexity=0.80, _maxConvexity=1.1, 
                        _minInertiaRatio=0.9, _maxInertiaRatio=1.0)
    #exibe(img, img_blobs, t2='Círculo')
    return (True, img_blobs) if len(keypoints)>0 else (False, img_blobs)

def isTriangulo(img):
    img_blobs, keypoints = detecta(img, _minCircularity=0.6, _maxCircularity=0.7, _minConvexity=0.50, _maxConvexity=1, 
                        _minInertiaRatio=0.5, _maxInertiaRatio=0.8)
    #exibe(img, img_blobs, t2='Triângulo')
    return (True, img_blobs) if len(keypoints)>0 else (False, img_blobs)

def detectaFormas(img):
    i, blur = filtros(img)
       
    quadrado, img_blobs = isQuadrado(i)
    circulo, img_blobs = isCirculo(i)
    triangulo, img_blobs = isTriangulo(i)
    s=''
    if circulo: 
        s='Circulo'
    elif quadrado: 
        s='Quadrado'
    elif triangulo: 
        s='Triangulo'
    else:
        s=''
    #forma imagem colorida para retornar
    i = cv2.merge([i,i,i])
    blur = cv2.merge([blur,blur,blur])
    return s, img_blobs, i, blur

def imprime_hist(img): 
    color = ('r','g','b')
    fig = plt.figure(figsize=(6,3))
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    
    # If we haven't already shown or saved the plot, then we need to draw the figure first...
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig.clf()
    plt.close()
    return data

def equaliza_imagem_colorida(img): #recebe em RGB
    frame = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2YCrCb)    # makes the blues image look real colored
    i_YCrCb = frame.copy()
    c0,c1,c2 = cv2.split(frame)
    c0 = cv2.equalizeHist(c0)
    frame = cv2.merge([c0,c1,c2])
    frame = cv2.cvtColor(frame, cv2.COLOR_YCrCb2RGB)    # makes the blues image look real colored
    return frame, i_YCrCb

def imprime_canais(img):
    (canalVermelho, canalVerde, canalAzul) = cv2.split(img)
    zeros = np.zeros(img.shape[:2], dtype = "uint8")
    #exibe(img,cv2.merge([zeros, zeros,canalVermelho]))
    #exibe(cv2.merge([zeros, canalVerde, zeros]),cv2.merge([canalAzul, zeros, zeros]))
    #exibe(img[:,:,0], img[:,:,2])
    r = cv2.merge([canalVermelho, zeros, zeros])
    g = cv2.merge([zeros, canalVerde, zeros])
    b = cv2.merge([zeros, zeros, canalAzul])
    return (r,g,b)
    
def detectaCor_e_Forma(img):
    original = img
    #eq, i_YCrCb = equaliza_imagem_colorida(img)
    #img=eq nao esta usando a imagem equalizada
    r,g,b = imprime_canais(img)
    #hist = cv2.resize(imprime_hist(img), (640,240))
    cor = encontraCor(img)    

    #print("Cor:", cor)
    if cor=="Amarelo" or cor=="Vermelho":
        forma, final, filtro, blur = detectaFormas(img[:,:,0])
    elif cor=="Azul":
        forma, final, filtro, blur = detectaFormas(img[:,:,2])
    else:
        forma = ''
        final = img
        filtro = img
        blur = img
    #exibe(img, final, t1=f, t2=str(forma+' '+cor))
    #print(f)

    ########################### Aciona LED da FORMA
#    if forma=="Circulo":
        #Aciona LED
        
#    elif cor=="Quadrado":
        #Aciona LED
        
#    elif cor=="Triangulo":
        #Aciona LED
        
        
    ########################### Aciona LED DA COR
#    if cor=="Amarelo":
        #Aciona LED
        
#    elif cor=="Vermelho":
        #Aciona LED
        
#    elif cor=="Azul":
        #Aciona LED
        

    #imprime a pilha completa
    #h1 = np.hstack([original, eq, i_YCrCb, blur])
    #h2 = np.hstack([filtro, final, hist])
    #h3 = np.hstack([r, g, b, img])
    #pilha = np.vstack([h1,h2,h3])

    #pilha reduzida só com final e RGB
    h1 = np.hstack([final, r])
    h2 = np.hstack([g, b])
    pilha = np.vstack([h1, h2])

    #h1 = np.hstack([original, 

    if forma != '' and cor != '': texto(pilha, forma+' '+cor, (10,250))

    #plt.imshow(pilha)
    return pilha
 
#####################################
## INICIO MAIN
#####################################
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print('Bem vindo(a) à ESTEIRA INTELIGENTE!')
print('\n\n\nSelecione a fonte da imagem:')
print('\n1. Câmera Pi')
print('2. Webcam')
op = int(input('\nOpção: '))
input('\n\n\nLimpe a área de leitura para calibrar e selecione 1 para iniciar...')


if op==1:
   try:
     from picamera.array import PiRGBArray
     from picamera import PiCamera
     # initialize the camera and grab a reference to the raw camera capture
     camera = PiCamera(sensor_mode=1) # se for 7 vai a 90 fps
     camera.resolution = (320, 240) #camera.resolution = (640, 480)
     camera.framerate = 32
     
     
     camera.zoom = (0.35, 0.45, 0.1, 0.1)
     time.sleep(3)
     camera.exposure_mode='off'
     camera.zoom = (0.0, 0.0, 1.0, 1.0)
     
     
     #print('analog_gain', camera.analog_gain, 'digital_gain', camera.digital_gain)
     
     camera.iso=400
     camera.exposure_compensation = 25
     
     #camera.awb_mode = 'off'
     camera.awb_mode = 'fluorescent'
     #camera.awb_gains = (1.5,1.5) # NORMALMENTE 0,9 A 1,9 MAS VAI DE 0 a 8
     
     #camera.drc_strength='off'
     camera.sharpness=100 # -100 a 100
     
     rawCapture = PiRGBArray(camera) #rawCapture = PiRGBArray(camera, size=(640, 480))
      
     # allow the camera to warmup
     time.sleep(0.1)
       
      
     # capture frames from the camera
     for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
       # grab the raw NumPy array representing the image, then initialize the timestamp
       # and occupied/unoccupied text
       #print('exposure_compensation', camera.exposure_compensation, 'iso', camera.iso, 'analog_g', camera.analog_gain, 'digital_g', camera.digital_gain)
       camera.awb_gains = (1.5,1.5) # NORMALMENTE 0,9 A 1,9 MAS VAI DE 0 a 8
     
       image = frame.array
       image = image.copy()
       
       image = image[::-1,::-1,::-1] # inverte BGR RGB e inverte linhas e colunas
       
       image = image[80:320,90:200]
       
       #image = image.copy()
       #image[10:60,:,:] = (255,0,0)
       i = detectaCor_e_Forma(image)
       # show the frame
       
       i = i[:,:,::-1] # inverte BGR RGB
       
       cv2.imshow("Frame", i)
       key = cv2.waitKey(1) & 0xFF
       # clear the stream in preparation for the next frame
       rawCapture.truncate(0)
       # if the `q` key was pressed, break from the loop
       if key == ord("q"):
         break
   except ImportError:
     print('Não esta rodando em um Raspberry')

elif op==2:
  # Se não tem picamera então captura da webcam
  vc = cv2.VideoCapture(0)
  vc.set(cv2.CAP_PROP_FRAME_WIDTH,320)  
  vc.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
  if vc.isOpened(): # try to get the first frame
     is_capturing, frame = vc.read()
  else:
     is_capturing = False  

  # calibragem
  img_clean = 0
  cont = 0
  while is_capturing:
    try: # Lookout for a keyboardInterrupt to stop the script
      is_capturing, frame = vc.read()   
      print('Calibrando... Aguarde!')
      time.sleep(0.2)
      if cont > 0:
          #img_clean = cv2.accumulate(frame, img_clean)
          img_clean = frame.copy()
      else:
          img_clean = frame.copy()
      cont+=1
      if cont>2:
        break
    
    except KeyboardInterrupt:
      vc.release()
    except:
      print('Erro!')
      vc.release()


  while is_capturing:
     try:    # Lookout for a keyboardInterrupt to stop the script
         is_capturing, frame = vc.read()   

         #sem usar calibragem
         #frame = frame - img_clean

         #frame = cv2.imread('circulo_amarelo.jpg')
         #frame = cv2.imread('triangulo_vermelho.jpg')
         #frame = cv2.imread('quadrado_azul.jpg')
         #frame = cv2.resize(frame[:,:,::-1], (320,240))

         i = detectaCor_e_Forma(frame[:,:,::-1]) #vai em formato RGB
         window_name = "Formas"
         #cv2.namedWindow(window_name, flags=cv2.WND_PROP_FULLSCREEN);

         cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
         cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


         cv2.imshow(window_name, i[:,:,::-1]) #converte para BGR para mostrar
         key = cv2.waitKey(1) & 0xFF
         # if the `q` key was pressed, break from the loop
         if key == ord("q"):
             break
     except KeyboardInterrupt:
         vc.release()
         cv2.destroyAllWindows()
else:
  print('Opção inválida!')


        
                                                    
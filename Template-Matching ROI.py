import cv2
import numpy as np
from scipy.signal import find_peaks
import os.path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time

def bildverarbeiten(frame, lower_color, higher_color):
    '''Bild wird in hsv umgewandelt, gefiltert und dann wird noch die CrossCorrelation-Methode angegeben'''
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_threshed = cv2.inRange(hsv_img, lower_color, higher_color)
    filtered_image = cv2.bitwise_and(frame, frame, mask=frame_threshed)
    method = 'cv2.TM_CCOEFF_NORMED'
    method = eval(method)
    return(hsv_img, frame_threshed, filtered_image, method)

def takeClosest(myList, myNumber, start, end):
    difference = abs(myNumber-myList[0])
    for i in range(start, end, 1):
        if abs(myList[i] - myNumber) < difference:
            difference = abs(myList[i] - myNumber)
            searched = i
    return searched

path_template = r'D:\...\NAME.jpg' #Pfad zum Vorlagenbild
path = r'D:\...\NAME.mp4' #Pfad zum Video
DIR = r'C:\...\NAME' #Ordner erstellen falls noch nicht vorhanden

for fname in os.listdir(DIR):  #so lösche ich alle Dateien die mit der Bezeichnung 'frame' beginnen
    if fname.startswith("frame"):
        os.remove(os.path.join(DIR, fname))

#farbfiltern
lower_color = np.array([20, 60, 20], np.uint8) #gelb
higher_color = np.array([60, 255, 255], np.uint8)

#Farbspektrum für blau
'''
#lower_color = np.array([80, 50, 50])  
#higher_color = np.array([130, 255, 255])
'''

value_list = []  # um höchstes Matching für perfekte Template-Größe zu ermitteln
coordinates_list = []  # Liste um aus den 3 tests den endgültig besten match zu ermitteln
change_log = []  # in diese Liste speichere ich die Indexes bei denen ein direction change stattfindet

height_list = []  # Höhe des Matches tracken
nummer = 0
half_window_rectangle = 50  # halbe Seitenlänge von ROI-Quadrat

#Falls die Bearbeitungszeit gestoppt werden soll, muss folgende Zeile eingeblendet werden
#t0 = time.time()

cap = cv2.VideoCapture(path)

while (True): #Jeden Frame laden
    ret, im = cap.read()
    nummer += 1
    try:
        if im.shape[0] > 4000:
            im = cv2.resize(im, (int(im.shape[1] / 7), int(im.shape[0] / 7)))
        elif im.shape[0] > 2000:
            im = cv2.resize(im, (int(im.shape[1] / 5), int(im.shape[0] / 5)))
        elif im.shape[0] > 1000:
            im = cv2.resize(im, (int(im.shape[1] / 3), int(im.shape[0] / 3)))

        #Falls das Bild um 90° gedreht werden soll, muss die folgende Zeile eingeblendet werden
        #im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        frame = im.copy()
    except:
        break
    hsv_img, frame_threshed, filtered_image, method = bildverarbeiten(frame, lower_color, higher_color)

    if nummer == 1:
        for x in range(6):  # perfekte Template-Size ermitteln
            template_size = 18 + x * 2
            template_old = cv2.imread(path_template, 1)
            template = cv2.resize(template_old, (template_size, template_size))  # Größe Template

            res = cv2.matchTemplate(filtered_image, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # gibt lokales minimum und maximum an
            coordinates_list.append(list(max_loc))
            value_list.append(max_val)
        best_template_size_index = value_list.index(max(value_list))
        coordinates = coordinates_list[best_template_size_index]
        template_size = 18 + best_template_size_index * 2  # um besten Pixelgröße zu ermitteln
        template = cv2.resize(template_old, (template_size, template_size))  # neues Template ist geboren

    #Maße des skalierten Frames
    height = frame.shape[1]
    width = frame.shape[0]

    if coordinates[1] < half_window_rectangle and coordinates[0] < half_window_rectangle: #Falls y-Achse und x-Achse negativ werden sollten
        case = 3
        frame = frame[0 : coordinates[1] + half_window_rectangle + template_size, 0 : coordinates[0] + half_window_rectangle + template_size]
    elif coordinates[1] < half_window_rectangle:  #Falls y-Achse negativ werden sollte
        case = 1
        frame = frame[0: height, coordinates[0] - half_window_rectangle: coordinates[0] + half_window_rectangle
                                + template_size]  #Falls der Wert negativ werden sollte, wird das Bild in y-Richtung nicht beschnitten
    elif coordinates[0] < half_window_rectangle:  #Falls x-Achse negativ werden sollte
        case = 2
        frame = frame[coordinates[1] - half_window_rectangle: coordinates[1] + half_window_rectangle
                    + template_size, 0:width]  #Falls negativ werden sollte beschneide Breite nicht
    else: #Falls nichts negativ wird
        case = 0
        frame = frame[
                coordinates[1] - half_window_rectangle: coordinates[1] + half_window_rectangle + template_size,
                coordinates[0] - half_window_rectangle: coordinates[0] + half_window_rectangle + template_size]

    hsv_img, frame_threshed, filtered_image, method = bildverarbeiten(frame, lower_color, higher_color)

    res = cv2.matchTemplate(filtered_image, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # In Absolutkoordinaten uebersetzen ... [0] --> x-Achse & [1] --> y-Achse
    if case == 1:
        coordinates[0] = coordinates[0] - half_window_rectangle + max_loc[0]
        coordinates[1] = max_loc[1]
    elif case == 0:
        coordinates[0] = coordinates[0] - half_window_rectangle + max_loc[0]
        coordinates[1] = coordinates[1] - half_window_rectangle + max_loc[1]
    elif case == 2:
        coordinates[0] = max_loc[0]
        coordinates[1] = coordinates[1] - half_window_rectangle + max_loc[1]
    else:
        coordinates[0] = max_loc[0]
        coordinates[1] = max_loc[1]

    height_list.append(coordinates[1])
    value_list.append(max_val)

#Falls die Frames während der Bearbeitung angezeigt werden sollen müssen die folgenden Zeilen eingeblendet werden
'''
    cv2.imshow("Matching Map", res)
    cv2.imshow("Original Image", frame)
    cv2.imshow("filtered Image", filtered_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''

#Falls die Bearbeitungszeit gestoppt werden soll, müssen folgende Zeilen eingeblendet werden
#t1 = time.time() #Bis zu dieser Zeile wird die benötigte Bearbeitungszeit gestoppt
#print("Bearbeitungszeit: ", t1-t0, " s")

x = range(0, len(height_list))
y = height_list
y_smoothed = savgol_filter(y, 51, 3) #Kurve glätten

average = sum(y_smoothed)/len(y_smoothed) #Mittelwert der Liste ermitteln
maxima_log = find_peaks(y_smoothed, distance=50, height=(average, max(y_smoothed)))[0] #finde Maxima

#Falls die Kurve ausgegeben werden soll, müssen die folgenden Zeilen eingeblendet werden
'''
plt.plot(x,y)
plt.plot(x,y_smoothed, color='red')
plt.xlabel('Frame')
plt.ylabel('Höhe')
plt.show()
'''

minima_log = []
for x in range(0,len(maxima_log)-1, 1):
    minima_log.append(int((maxima_log[x] + maxima_log[x+1])/2))

zwischenpunkte_log = []

for x in range(0,len(maxima_log)-1, 1):
    maximum = y_smoothed[maxima_log[x + 1]]
    minimum = y_smoothed[minima_log[x]]
    searched_angle = int((maximum + minimum) / 2)
    try:
        zwischenpunkt = takeClosest(y_smoothed, searched_angle, minima_log[x] ,maxima_log[x+1]) #um Bild mit richtigem Winkel
    except:
        break
    zwischenpunkte_log.append(zwischenpunkt)
for x in range(0,len(maxima_log)-1, 1):
    maximum = y_smoothed[maxima_log[x]]
    minimum = y_smoothed[minima_log[x]]
    searched_angle = int((maximum + minimum) / 2)
    try:
        zwischenpunkt = takeClosest(y_smoothed, searched_angle, maxima_log[x], minima_log[x])
    except:
        break
    zwischenpunkte_log.append(zwischenpunkt)

writing_list = [] #Liste mit Framenummern dessen zugehoerige Frames abgespeichert werden sollen

for x in zwischenpunkte_log:
    writing_list.append(x)
for y in maxima_log:
    writing_list.append(y)
for z in minima_log:
    writing_list.append(z)
writing_list.sort() #Framenummern sortieren

for i in writing_list: #Bilder für chosen_pics aussuchen
    cap = cv2.VideoCapture(path)
    cap.set(1, i)  #i steht für Framenummer welche Geladen werden soll
    ret, img = cap.read()
    #wenn man die abzuspeichernden Bilder skalieren möchte, müssen folgende ausgeklammerte Zeilen eingeblendet werden
    '''
    if img.shape[0] > 4000:
        img = cv2.resize(img, (int(img.shape[1] / 7), int(img.shape[0] / 7)))
    elif img.shape[0] > 2000:
        img = cv2.resize(img, (int(img.shape[1] / 5), int(img.shape[0] / 5)))
    elif img.shape[0] > 1000:
        img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
    # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    '''
    cv2.imwrite(r'C:\...\frame{}.jpg'.format(i), img) #abspeichern

import pygame # Digituak idazteko
import numpy as np
import time
import pandas as pd
from sklearn.svm import SVC # Jada SVM inplementatutako pythoneko pakete bat
from random import randint
import pickle
import matplotlib.pyplot as plt
import pygame_gui

# Parametroak (alda daitezke):
grid_size = 20  # Karratu bakoitzaren tamaina pixeletan (minimoa 20)

# Koloreak
WHITE = (238,238,210)
BLACK = (186,202,68)
GRID_COLOR = (200, 200, 200) # Karratuen arteko kolorea eta pantaila atzeko kolorea
TEXT_COLOR = (0,0,0)
BUTTON_COLOR = (200, 200, 200)
GREEN = (0, 255, 0)
RED =(255,0,0)

# Hemendik aurrerakoa ez aldatu













# -------------------------------------------------------------------------------
# -----------------------------DATU BASEA INPORTATU------------------------------
# -------------------------------------------------------------------------------
entrenamendu_datuak = pd.read_csv("mnist_train.csv")
testeatzeko_datuak = pd.read_csv("mnist_test.csv")
# print(entrenamendu_datuak.head())
X_entrenamendu = entrenamendu_datuak.iloc[:,1:]
Y_entrenamendu = entrenamendu_datuak["label"]

X_test = testeatzeko_datuak.iloc[:,1:]
Y_test = testeatzeko_datuak["label"]

zutabeen_izenak = X_entrenamendu.columns.tolist()

# -------------------------------------------------------------------------------
# -------------------------------MODELOA INPORTATU-------------------------------
# -------------------------------------------------------------------------------

modeloa = pickle.load(open("SkLearn_SVC_model_C_4.pkl","rb"))

# Honek ez badu funtzionatzen, ondorengo kodea exekutatu (ctrl + k + u, deskomentatzeko): 

# modeloa = SVC(C = 4)
# hasierako_denbora = time.time()
# modeloa.fit(X_entrenamendu.values,Y_entrenamendu) # .values egiten dugu horrela zutabeen
# # izenak ez dira beharrezkoak aurresaterako garaian
# amaierako_denbora = time.time()
# print(f"Entrenatzeko beharrezko denbora: {amaierako_denbora-hasierako_denbora}s")













# -------------------------------------------------------------------------------
# ----------------------------APLIKAZIO INTERAKTIBOA-----------------------------
# -------------------------------------------------------------------------------

# Pantailaren tamainiako aldagai orokorrak:
n = 28

grid_width, grid_height = n * grid_size, n * grid_size
width = grid_width + 300 # Instrukzioak / botoiak gehitzeko
height = grid_height


# Hainbat funtzio auxiliar:
def kuadrikula_marraztu(kuadrikula,screen):
    """
    grid = 28 x 28 tamainiako matrize bat: matrizearen elementuak 0 - 255 tarteko zenbakiak
    izango dira, 28 x 28 pixeleko pantaila baten gris-eskalak izanik
    """
    screen.fill(WHITE)
    for i in range(n):
        for j in range(n):
            balioa = kuadrikula[i][j]
            pygame.draw.rect(screen, (balioa,balioa,balioa), (j * grid_size, i * grid_size, grid_size, grid_size))
            pygame.draw.rect(screen, GRID_COLOR, (j * grid_size, i * grid_size, grid_size, grid_size), 1) # 1 Balioa da
            # karratuaren zenbat betetzen den definitzeko, 0 da "default" balioa eta karratu osoa betetzen du.

def digituak_marraztu(zenbakia,errenkada,zutabea):
    # Klikatutako karratua
    if 0 <= errenkada < n and 0 <= zutabea < n:
        zenbakia[errenkada][zutabea] = 255
    inguruneak = [(1,0),(-1,0),(0,1),(0,-1)]
    for dx,dy in inguruneak:
        zut = zutabea + dx
        errenk = errenkada + dy
        if 0 <= errenk < n and 0 <= zut < n and zenbakia[errenk][zut] <= 200:
            zenbakia[errenk][zut] = randint(150,220)

def digituak_garbitu(zenbakia,errenkada,zutabea):
    # Klikatutako karratua
    if 0 <= errenkada < n and 0 <= zutabea < n:
        zenbakia[errenkada][zutabea] = 0
    inguruneak = [(1,0),(-1,0),(0,1),(0,-1)]
    for dx,dy in inguruneak:
        zut = zutabea + dx
        errenk = errenkada + dy
        if 0 <= errenk < n and 0 <= zut < n:
            zenbakia[errenk][zut] = 0

def modeloa_sortu(puntuak,izenak,kernel_mota="kernel gaussiarra",koefizientea = 4):
    X = np.array(puntuak)
    Y = np.array(izenak)
    mean = np.mean(X,0)
    sd = np.std(X,0)
    X = (X-mean)/sd

    X[:,1] = -X[:,1]
    
    if kernel_mota == "kernel gaussiarra":
        model = SVC(C = koefizientea, kernel = "rbf")
        k = "gaussian"
    elif kernel_mota == "kernel polinomiala":
        model = SVC(C = koefizientea, kernel = "poly")
        k = "polynomial"
    elif kernel_mota == "kernel lineala":
        model = SVC(C = koefizientea, kernel = "linear")
        k = "linear"

    model.fit(X,Y)

    # Step 2: Generate mesh grid
    # Define the range of x and y values
    x_min, x_max = np.min(X[:, 0]) - 0.2, np.max(X[:, 0]) + 0.2
    y_min, y_max = np.min(X[:, 1]) - 0.2, np.max(X[:, 1]) + 0.2

    # Create a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))

    # Step 3: Predict class labels for all points on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Step 4: Plot decision surface
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, edgecolors='k')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    plt.title(f'Decision surface\n kernel = {k}, C = {koefizientea}, Model score = {round(model.score(X,Y),3)}')
    plt.show()

    return model

# Botoien funtzionalitate orokorra         
class Botoia:
    def __init__(self, x, y, width, height, color, text, font_size = 24):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.text = text
        self.font = pygame.font.Font(None, font_size)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        text_surface = self.font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

# Menu desberdiñak: 
def adibideak_ikusi():
    """
    Programa honek MNIST datu-baseko balioak ikusteko balio du
    """
    # Pygame hasi
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("View some examples from the MNIST database")

    # Beharrezko hainbat aldagai
    running = True
    zenbagarren_adibidea = 0

    # Textuak idazteko beharrezkoa:
    izenburua = pygame.font.Font(None, 30)
    digitua = pygame.font.Font(None, 300)
    instrukzioak = pygame.font.Font(None,20)

    # Botoia:
    botoia_hurrengo_adibidea = Botoia(width - 230, 325, 170, 40, BUTTON_COLOR, "Next example")
    botoia_itzuli_menura = Botoia(width - 230, height - 50, 170, 40, BUTTON_COLOR, "Go to menu")

    # Loop orokorra
    while running:
        for event in pygame.event.get():
            keys = pygame.key.get_pressed()
            if event.type == pygame.QUIT: # Exekutatzeaz bukatzeko
                running = False
                zer_egin = "amaitu"
            elif keys[pygame.K_RETURN]: # Enter botoia sakatzerakoan
                zenbagarren_adibidea = randint(0,1000)
            elif event.type == pygame.MOUSEBUTTONDOWN: # Xaguaren botoiren bat sakatzen bada
                if event.button == 1: # Ezkerreko botoia sakatzen bada
                    if botoia_itzuli_menura.rect.collidepoint(event.pos):
                        running = False
                        zer_egin = "menua_ireki"
                    elif botoia_hurrengo_adibidea.rect.collidepoint(event.pos):
                        zenbagarren_adibidea = randint(0,1000)

        # while buklearen amaieran, pantaila marraztu, textua idatzi eta aktualizatzeko
        kuadrikula_marraztu(np.array(X_entrenamendu.iloc[zenbagarren_adibidea,:]).reshape(n,n),screen)
        screen.blit(izenburua.render("Examples from MNIST", True, TEXT_COLOR),(width - 295, 30))
        screen.blit(digitua.render(str(Y_entrenamendu[zenbagarren_adibidea]), True, BLACK),(width - 205, 130))
        screen.blit(instrukzioak.render("Click enter to see the next example", True, TEXT_COLOR),(width - 295, 55))
        botoia_itzuli_menura.draw(screen)
        botoia_hurrengo_adibidea.draw(screen)
        pygame.display.flip()
        
    return zer_egin

def predezitu():
    """
    Programa honek xaguarekin digituak idazteko balio du, gero modeloak predezitzeko
    """
    # Pygame hasi
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Digit recognition")

    # Beharrezko hainbat aldagai
    running = True
    marrazten = False
    garbitzen = False
    zenbakia = np.zeros((n,n),dtype=int)
    predikzioa = "1"

    # Textuak idazteko beharrezkoa:
    izenburua = pygame.font.Font(None, 37)
    digitua = pygame.font.Font(None, 300)
    instrukzioak = pygame.font.Font(None,20)

    # Botoia:
    botoia_garbitu = Botoia(width - 230, 325, 170, 40, BUTTON_COLOR, "Clean screen")
    botoia_itzuli_menura = Botoia(width - 230, height - 50, 170, 40, BUTTON_COLOR, "Go to menu")

    # Loop orokorra
    while running:
        if np.array_equal(zenbakia,np.zeros((n,n),dtype=int)):
            predikzioa = "?"
        else:
            predikzioa = modeloa.predict(zenbakia.reshape((1,n**2)))[0]

        for event in pygame.event.get():
            keys = pygame.key.get_pressed()
            if event.type == pygame.QUIT: # Exekutatzeaz bukatzeko
                running = False
                zer_egin = "amaitu"
            elif keys[pygame.K_RETURN]: # Enter botoia sakatzerakoan
                zenbakia = np.zeros((n,n),dtype=int)
            elif event.type == pygame.MOUSEBUTTONDOWN: # Xaguaren botoiren bat sakatzen bada
                if event.button == 1: # Ezkerreko botoia sakatzen bada
                    if botoia_itzuli_menura.rect.collidepoint(event.pos):
                        running = False
                        zer_egin = "menua_ireki"
                    elif botoia_garbitu.rect.collidepoint(event.pos):
                        zenbakia = np.zeros((n,n))
                    else:    
                        marrazten = True
                        x,y = event.pos
                        zutabea = x // grid_size
                        errenkada = y // grid_size
                        digituak_marraztu(zenbakia,errenkada,zutabea)
                elif event.button == 3: # Eskubiko botoia sakatzen bada
                    garbitzen = True
                    x,y = event.pos
                    zutabea = x // grid_size
                    errenkada = y // grid_size
                    digituak_garbitu(zenbakia,errenkada,zutabea)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Botón izquierdo del ratón
                    marrazten = False
                if event.button == 3:
                    garbitzen = False
            elif event.type == pygame.MOUSEMOTION: # Xagua mugitzen bada
                if marrazten:
                    x,y = event.pos
                    zutabea = x // grid_size
                    errenkada = y // grid_size
                    digituak_marraztu(zenbakia,errenkada,zutabea)
                elif garbitzen:
                    x,y = event.pos
                    zutabea = x // grid_size
                    errenkada = y // grid_size
                    digituak_garbitu(zenbakia,errenkada,zutabea)

        # while buklearen amaieran, pantaila marraztu, textua idatzi eta aktualizatzeko
        kuadrikula_marraztu(zenbakia,screen)
        screen.blit(izenburua.render("Model's prediction", True, TEXT_COLOR),(width - 290, 20))
        screen.blit(digitua.render(str(predikzioa), True, BLACK),(width - 215, 80))
        screen.blit(instrukzioak.render("Click 'enter' to clean the screen", True, TEXT_COLOR),(width - 295, 55))
        botoia_itzuli_menura.draw(screen)
        botoia_garbitu.draw(screen)
        pygame.display.flip()
        
    return zer_egin

def SVM_bisuala():
    """
    Programa honek xaguarekin laginak sortzeko balio du ondoren bisualki SVM-ren outputa ikusteko
    """
    # Pygame hasi
    screen = pygame.display.set_mode((width, height))
    grafikoa = pygame.Surface((grid_width,grid_height))
    pygame.display.set_caption("SVM visualization")
    screen.fill(WHITE)
    grafikoa.fill((0,0,0))
    botoien_distantzia = height // 10

    # Modeloaren koefizientea aldatzeko barra sortzeko:
    manager = pygame_gui.UIManager((width, height))
    slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((width - 230, height - 50 - 3 * botoien_distantzia - 30), (170, 20)),
        start_value=0,  # Valor inicial del deslizador
        value_range=(-4, 5),  # Rango de valores del deslizador
        manager=manager)

    # Beharrezko hainbat aldagai
    running = True
    marrazten = False
    puntu_berdeak = True
    berdeak = []
    gorriak = []
    puntu_guztiak = []
    izenak = []
    i = 0
    j = 0
    clock = pygame.time.Clock()
    delta_time = clock.tick(60) / 1000.0
    kernel_motak = ["kernel gaussiarra","kernel polinomiala","kernel lineala"]
    modeloa = "kernel gaussiarra"
    koefizientea = 1

    # Textuak idazteko beharrezkoa:
    izenburua = pygame.font.Font(None, 37)
    instrukzioak = pygame.font.Font(None,20)
    puntuak = pygame.font.Font(None,30)

    # Botoia:
    botoia_garbitu = Botoia(width - 230, height - 50 - 1 * botoien_distantzia, 170, 40, BUTTON_COLOR, "Clean screen")
    botoia_itzuli_menura = Botoia(width - 230, height - 50, 170, 40, BUTTON_COLOR, "Go to menu")
    botoia_modeloa_sortu = Botoia(width - 230, height - 50 - 2 * botoien_distantzia, 170, 40, BUTTON_COLOR, "Generate model")
    botoia_kernel_mota = Botoia(width - 230, height - 50 - 3 * botoien_distantzia, 170, 40, BUTTON_COLOR, "Kernel type")

    # Loop orokorra
    while running:
        for event in pygame.event.get():
            manager.process_events(event)
            keys = pygame.key.get_pressed()
            if event.type == pygame.QUIT: # Exekutatzeaz bukatzeko
                running = False
                zer_egin = "amaitu"
            elif keys[pygame.K_RETURN]: # Enter botoia sakatzerakoan
                if not(len(gorriak) == 0 or len(berdeak) == 0):
                    modeloa_sortu(puntu_guztiak,izenak,modeloa,koefizientea)

            elif event.type == pygame.MOUSEBUTTONDOWN: # Xaguaren botoiren bat sakatzen bada
                if event.button == 1: # Ezkerreko botoia sakatzen bada
                    if botoia_itzuli_menura.rect.collidepoint(event.pos):
                        running = False
                        zer_egin = "menua_ireki"
                    elif botoia_garbitu.rect.collidepoint(event.pos):
                        grafikoa.fill((0,0,0))
                        berdeak = []
                        gorriak = []
                        izenak = []
                        puntu_guztiak = []
                    elif botoia_kernel_mota.rect.collidepoint(event.pos):
                        j += 1
                        modeloa = kernel_motak[j % 3]
                    
                    elif botoia_modeloa_sortu.rect.collidepoint(event.pos):
                        if not(len(gorriak) == 0 or len(berdeak) == 0):
                            modeloa_sortu(puntu_guztiak,izenak,modeloa,koefizientea)

                    else: # Puntu berdeak marrazteko   
                        marrazten = True
                        puntu_berdeak = True
                        x,y = event.pos
                        if 0 <= x < grid_width and 0 <= y < grid_height:
                            berdeak.append(event.pos)
                            puntu_guztiak.append(event.pos)
                            izenak.append(1)
                elif event.button == 3: # Eskubiko botoia sakatzen bada puntu gorriak marraztu
                    marrazten = True
                    puntu_berdeak = False
                    x,y = event.pos
                    if 0 <= x < grid_width and 0 <= y < grid_height:
                        gorriak.append(event.pos)
                        puntu_guztiak.append(event.pos)
                        izenak.append(-1)

            elif event.type == pygame.MOUSEBUTTONUP:
                marrazten = False
            elif event.type == pygame.MOUSEMOTION: # Xagua mugitzen bada
                if marrazten and puntu_berdeak and i % 20 == 0: # i sartzen dugu "cadencia" bat egoteko puntuen gehikuntzan
                    x,y = event.pos
                    if 0 <= x < grid_width and 0 <= y < grid_height:
                        berdeak.append(event.pos)
                        puntu_guztiak.append(event.pos)
                        izenak.append(1)
                if marrazten and not puntu_berdeak and i % 20 == 0: 
                    x,y = event.pos
                    if 0 <= x < grid_width and 0 <= y < grid_height:
                        gorriak.append(event.pos)
                        puntu_guztiak.append(event.pos)
                        izenak.append(-1)  
                i += 1

        # while buklearen amaieran, pantaila marraztu, textua idatzi eta aktualizatzeko
        screen.fill(WHITE)
        screen.blit(grafikoa,(0,0))

        # Modeloaren koefizientea lortu
        manager.update(delta_time)
        manager.draw_ui(screen)
        koefizientea = round(2**(slider.get_current_value()),3)

        # Puntuak marraztu
        for punto in berdeak:
            pygame.draw.circle(screen, GREEN, punto, 5)
        for punto in gorriak:
            pygame.draw.circle(screen, RED, punto, 5)
        
        screen.blit(izenburua.render("SVM model generator", True, BLACK),(width - 290, 20))
        screen.blit(instrukzioak.render("- Draw a sample and generate models", True, TEXT_COLOR),(width - 295, 55))
        screen.blit(instrukzioak.render("- Click 'enter' to generate model", True, TEXT_COLOR),(width - 295, 75))
        screen.blit(instrukzioak.render("- Be aware that the number of green and", True, TEXT_COLOR),(width - 295, 95))
        screen.blit(instrukzioak.render("  red points should be similar, if not,", True, TEXT_COLOR),(width - 295, 115))
        screen.blit(instrukzioak.render("  model might not be created", True, TEXT_COLOR),(width - 295, 135))
        screen.blit(puntuak.render(f"Amount of green points = {len(berdeak)}", True, (0,100,0)),(width - 295, 175))
        screen.blit(puntuak.render(f"Amount of red points = {len(gorriak)}", True, (100,0,0)),(width - 295, 195))
        screen.blit(puntuak.render(f"Model type:", True, TEXT_COLOR),(width - 295, 215))
        screen.blit(puntuak.render(modeloa, True, TEXT_COLOR),(width - 270, 235))
        screen.blit(puntuak.render(f"C = {koefizientea}", True, TEXT_COLOR),(width - 270, 255))
        botoia_itzuli_menura.draw(screen)
        botoia_garbitu.draw(screen)
        botoia_modeloa_sortu.draw(screen)
        botoia_kernel_mota.draw(screen)

        pygame.display.flip()
        

    return zer_egin

def menua():
    """
    Programa hau menua exekutatzeko da
    """
    # Pygame hasi
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Interactive SVM with MNIST dataset")

    # Irudiak:
    botoien_dist = height//10
    irudia = pygame.image.load("Menu_figure.png") 
    irudia = pygame.transform.scale(irudia, (botoien_dist * 5.5, botoien_dist*4))
    irudia_rect = irudia.get_rect()
    irudia_rect.center = (width // 2, height // 2 - botoien_dist* 1.3)

    # Beharrezko aldagaia
    running = True

    # Textuak idazteko beharrezkoa:
    izenburua = pygame.font.Font(None, 80)

    # Botoia:
    botoia_adibideak_ikusi = Botoia(width//2-350//2, height // 2 + botoien_dist * 1, 350, 40, BUTTON_COLOR, "See examples from MNIST")
    botoia_digituak_predezitu = Botoia(width//2-350//2, height // 2 + botoien_dist * 2 , 350, 40, BUTTON_COLOR, "Draw digits to be recognized")
    botoia_SVC_bisualizadorea = Botoia(width//2-350//2, height // 2 + botoien_dist * 3 , 350, 40, BUTTON_COLOR, "Generate samples and models in 2D")
    botoia_irten = Botoia(width//2-350//2, height // 2 + botoien_dist * 4, 350, 40, BUTTON_COLOR, "Exit")

    # Loop orokorra
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # Exekutatzeaz bukatzeko
                running = False
                zer_egin = "amaitu"
            elif event.type == pygame.MOUSEBUTTONDOWN: # Xaguaren botoiren bat sakatzen bada
                if event.button == 1: # Ezkerreko botoia sakatzen bada
                    if botoia_adibideak_ikusi.rect.collidepoint(event.pos):
                        running = False
                        zer_egin = "adibideak_ireki"
                    elif botoia_digituak_predezitu.rect.collidepoint(event.pos):
                        running = False
                        zer_egin = "predezitu_ireki"
                    elif botoia_irten.rect.collidepoint(event.pos):
                        running = False
                        zer_egin = "amaitu"
                    elif botoia_SVC_bisualizadorea.rect.collidepoint(event.pos):
                        running = False
                        zer_egin = "SVM_bisuala"

        # while buklearen amaieran, pantaila marraztu, textua idatzi eta aktualizatzeko
        screen.fill(WHITE)
        screen.blit(izenburua.render("Interactive MNIST and SVM", True, BLACK),(width//2 - 365, 20))
        botoia_adibideak_ikusi.draw(screen)
        botoia_digituak_predezitu.draw(screen)
        botoia_irten.draw(screen)
        botoia_SVC_bisualizadorea.draw(screen)
        screen.blit(irudia, irudia_rect)
        pygame.display.flip()

    return zer_egin
    
def main():
    # Beharrezko aldagaiak:
    zer_egin = "menua_ireki"

    # Pygame hasi
    pygame.init()

    while zer_egin != "amaitu":

        if zer_egin == "menua_ireki":
            zer_egin = menua()

        elif zer_egin == "adibideak_ireki":
            zer_egin = adibideak_ikusi()

        elif zer_egin == "predezitu_ireki":
            zer_egin = predezitu()
        
        elif zer_egin == "SVM_bisuala":
            zer_egin = SVM_bisuala()

    # Pygame amaitu
    pygame.quit()















if __name__ == "__main__":
    main()
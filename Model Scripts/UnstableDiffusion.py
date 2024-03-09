## GUI STUFF ##
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QSplashScreen
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap
import os
import datetime

## DIFFUSION STUFF ##
import model_loader
import pipeline
from PIL import Image
from PIL import ImageQt
from transformers import CLIPTokenizer
import torch


class Ui_MainWindow(object):

    def __init__(self):
        self.DEVICE = "cpu"
        self.ALLOW_CUDA = False
        self.ALLOW_MPS = False

        if torch.cuda.is_available() and self.ALLOW_CUDA:
            self.DEVICE = "cuda"
        elif (torch.backends.mps.is_built() or torch.backends.mps.is_available()) and self.ALLOW_MPS:
            self.DEVICE = "mps"
        print(f"Using device: {self.DEVICE}")

        self.tokenizer = CLIPTokenizer("../Data/tokenizer_vocab.json", merges_file="../Data/tokenizer_merges.txt")
        self.model_file = "../Data/v1-5-pruned-emaonly.ckpt"
        self.models = model_loader.preload_models_from_standard_weights(self.model_file, self.DEVICE)

        ## TEXT TO IMAGE

        # prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
        # prompt = "put sunglass on the eye"
        self.uncond_prompt = ""  # Also known as negative prompt
        self.do_cfg = True
        self.cfg_scale = 8  # min: 1, max: 14

        ## IMAGE TO IMAGE

        self.input_image = None
        # Comment to disable image to image
        self.image_path = "../Images/Rick.jpg"
        # input_image = Image.open(image_path)
        # Higher values means more noise will be added to the input image, so the result will further from the input image.
        # Lower values means less noise is added to the input image, so output will be closer to the input image.
        self.strength = 0.9

        ## SAMPLER

        self.sampler = "ddpm"
        self.num_inference_steps = 50
        self.seed = 42

        self.pixmap = None
        ####################################################################################################################

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1334, 720)
        MainWindow.setMinimumSize(QtCore.QSize(1280, 720))
        MainWindow.setMaximumSize(QtCore.QSize(1334, 720))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("logo.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 20, 1321, 80))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(10, 30, 141, 16))
        self.label.setObjectName("label")
        self.prompt = QtWidgets.QTextEdit(self.frame)
        self.prompt.setGeometry(QtCore.QRect(170, 0, 841, 81))
        self.prompt.setObjectName("prompt")
        self.usecuda = QtWidgets.QCheckBox(self.frame)
        self.usecuda.setGeometry(QtCore.QRect(1020, 20, 121, 41))
        self.usecuda.setObjectName("usecuda")
        self.generate = QtWidgets.QPushButton(self.frame)
        self.generate.setGeometry(QtCore.QRect(1170, 30, 101, 23))
        self.generate.setObjectName("generate")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(10, 110, 1331, 151))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.progress = QtWidgets.QTextBrowser(self.frame_2)
        self.progress.setGeometry(QtCore.QRect(25, 10, 1281, 131))
        self.progress.setObjectName("progress")
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(10, 270, 1331, 381))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.save = QtWidgets.QPushButton(self.frame_3)
        self.save.setGeometry(QtCore.QRect(580, 340, 104, 28))
        self.save.setObjectName("save")
        self.photo = QtWidgets.QLabel(self.frame_3)
        self.photo.setGeometry(QtCore.QRect(470, 10, 320, 320))
        self.photo.setMaximumSize(QtCore.QSize(512, 512))
        self.photo.setAutoFillBackground(False)
        self.photo.setText("")
        # self.photo.setPixmap(QtGui.QPixmap("logo.png"))
        self.photo.setScaledContents(True)
        self.photo.setObjectName("photo")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1334, 25))
        self.menubar.setObjectName("menubar")
        self.menuAbout = QtWidgets.QMenu(self.menubar)
        self.menuAbout.setObjectName("menuAbout")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuAbout.menuAction())

        ############### modified shit goes brrrrrrrrrrr ######################################
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.generate.clicked.connect(self.generate_photo)
        self.save.clicked.connect(self.save_image)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Unstable Diffusion"))
        self.label.setText(_translate("MainWindow", "Enter Prompt :"))
        self.usecuda.setText(_translate("MainWindow", "Use CUDA"))
        self.generate.setText(_translate("MainWindow", "Generate"))
        self.save.setText(_translate("MainWindow", "Save Image"))
        self.menuAbout.setTitle(_translate("MainWindow", "About"))

    def generate_photo(self):
        iscuda = self.usecuda.isChecked()
        if iscuda:
            self.DEVICE = "cuda"
        else:
            self.DEVICE = "cpu"

        prompt = self.prompt.toPlainText()
        if prompt.isspace() or len(prompt) == 0:
            self.prompt.setText("madafaka at least write somthing !")
            return None

        output_image = pipeline.generate(
            prompt=prompt,
            uncond_prompt=self.uncond_prompt,
            input_image=self.input_image,
            strength=self.strength,
            do_cfg=self.do_cfg,
            cfg_scale=self.cfg_scale,
            sampler_name=self.sampler,
            n_inference_steps=self.num_inference_steps,
            seed=self.seed,
            models=self.models,
            device=self.DEVICE,
            idle_device="cpu",
            tokenizer=self.tokenizer,
        )

        image = Image.fromarray(output_image)
        q_image = ImageQt.ImageQt(image)  # Convert PIL Image to QImage
        self.pixmap = QPixmap.fromImage(q_image)  # Convert QImage to QPixmap

        # Set the pixmap to self.photo
        self.photo.setPixmap(self.pixmap)

    def save_image(self):
        file_path = get_unstable_diffusion_directory()
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        image_filename = f"image_{formatted_time}.jpg"

        if file_path:
            pixmap = QPixmap("logo.png")  # Load the image
            image_path = os.path.join(file_path, image_filename)  # Construct full image path
            print(image_path)
            pixmap.save(image_path)  # Save the image


def get_unstable_diffusion_directory():
    # Get the path to the desktop directory
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

    # Create the full path to the "Unstable Diffusion" directory
    unstable_diffusion_path = os.path.join(desktop_path, "Unstable Diffusion")

    # Check if the directory exists
    if not os.path.exists(unstable_diffusion_path):
        # If it doesn't exist, create it
        os.makedirs(unstable_diffusion_path)

    return unstable_diffusion_path


class SplashScreen(QSplashScreen):
    def __init__(self, pixmaps):
        super().__init__(pixmaps, flags=Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setFixedSize(pixmaps.size())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # Load the image for the splash screen
    pixmaps = QPixmap("./logo.png")
    splash = SplashScreen(pixmaps)

    # Show the splash screen
    splash.show()

    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    splash.close()
    sys.exit(app.exec_())

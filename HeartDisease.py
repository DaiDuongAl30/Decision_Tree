# Form implementation generated from reading ui file 'HeartDisease.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.



import pandas as pd
import subprocess
import re
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QFileDialog
from collections import Counter
import math
from sklearn.model_selection import train_test_split
from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(969, 554)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.predictButt = QtWidgets.QPushButton(parent=self.centralwidget)
        self.predictButt.setGeometry(QtCore.QRect(440, 270, 75, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.predictButt.setFont(font)
        self.predictButt.setObjectName("predictButt")
        self.sex = QtWidgets.QComboBox(parent=self.centralwidget)
        self.sex.setGeometry(QtCore.QRect(160, 110, 69, 22))
        self.sex.setObjectName("sex")
        self.sex.addItem("")
        self.sex.addItem("")
        self.age = QtWidgets.QComboBox(parent=self.centralwidget)
        self.age.setGeometry(QtCore.QRect(160, 80, 69, 22))
        self.age.setEditable(False)
        self.age.setObjectName("age")
        self.age.addItem("")
        self.age.addItem("")
        self.age.addItem("")
        self.chestPain = QtWidgets.QComboBox(parent=self.centralwidget)
        self.chestPain.setGeometry(QtCore.QRect(160, 140, 69, 22))
        self.chestPain.setObjectName("chestPain")
        self.chestPain.addItem("")
        self.chestPain.addItem("")
        self.chestPain.addItem("")
        self.chestPain.addItem("")
        self.restRB = QtWidgets.QComboBox(parent=self.centralwidget)
        self.restRB.setGeometry(QtCore.QRect(160, 170, 69, 22))
        self.restRB.setObjectName("restRB")
        self.restRB.addItem("")
        self.restRB.addItem("")
        self.restRB.addItem("")
        self.chol = QtWidgets.QComboBox(parent=self.centralwidget)
        self.chol.setGeometry(QtCore.QRect(160, 200, 69, 22))
        self.chol.setObjectName("chol")
        self.chol.addItem("")
        self.chol.addItem("")
        self.chol.addItem("")
        self.fastBS = QtWidgets.QComboBox(parent=self.centralwidget)
        self.fastBS.setGeometry(QtCore.QRect(440, 80, 69, 22))
        self.fastBS.setObjectName("fastBS")
        self.fastBS.addItem("")
        self.fastBS.addItem("")
        self.resElec = QtWidgets.QComboBox(parent=self.centralwidget)
        self.resElec.setGeometry(QtCore.QRect(770, 170, 191, 22))
        self.resElec.setObjectName("resElec")
        self.resElec.addItem("")
        self.resElec.addItem("")
        self.resElec.addItem("")
        self.thalach = QtWidgets.QComboBox(parent=self.centralwidget)
        self.thalach.setGeometry(QtCore.QRect(440, 110, 69, 22))
        self.thalach.setObjectName("thalach")
        self.thalach.addItem("")
        self.thalach.addItem("")
        self.thalach.addItem("")
        self.exang = QtWidgets.QComboBox(parent=self.centralwidget)
        self.exang.setGeometry(QtCore.QRect(440, 140, 69, 22))
        self.exang.setObjectName("exang")
        self.exang.addItem("")
        self.exang.addItem("")
        self.oldpeak = QtWidgets.QComboBox(parent=self.centralwidget)
        self.oldpeak.setGeometry(QtCore.QRect(440, 170, 69, 22))
        self.oldpeak.setObjectName("oldpeak")
        self.oldpeak.addItem("")
        self.oldpeak.addItem("")
        self.oldpeak.addItem("")
        self.slope = QtWidgets.QComboBox(parent=self.centralwidget)
        self.slope.setGeometry(QtCore.QRect(770, 110, 121, 22))
        self.slope.setObjectName("slope")
        self.slope.addItem("")
        self.slope.addItem("")
        self.slope.addItem("")
        self.ca = QtWidgets.QComboBox(parent=self.centralwidget)
        self.ca.setGeometry(QtCore.QRect(770, 80, 69, 22))
        self.ca.setObjectName("ca")
        self.ca.addItem("")
        self.ca.addItem("")
        self.ca.addItem("")
        self.ca.addItem("")
        self.thal = QtWidgets.QComboBox(parent=self.centralwidget)
        self.thal.setGeometry(QtCore.QRect(770, 140, 121, 22))
        self.thal.setObjectName("thal")
        self.thal.addItem("")
        self.thal.addItem("")
        self.thal.addItem("")
        self.lineEdit = QtWidgets.QLineEdit(parent=self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(420, 320, 113, 31))
        self.lineEdit.setReadOnly(True)
        self.lineEdit.setObjectName("lineEdit")
        self.label = QtWidgets.QLabel(parent=self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 80, 47, 13))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 110, 47, 13))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(40, 140, 91, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(40, 170, 121, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(40, 200, 101, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(280, 80, 111, 20))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(560, 170, 211, 20))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(280, 110, 151, 16))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(280, 140, 141, 16))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(280, 170, 101, 16))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(560, 110, 81, 16))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(560, 80, 141, 16))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(560, 140, 121, 16))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(310, 20, 341, 20))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_14.setFont(font)
        self.label_14.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(340, 330, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.chooseRules = QtWidgets.QPushButton(parent=self.centralwidget)
        self.chooseRules.setGeometry(QtCore.QRect(560, 220, 81, 23))
        self.chooseRules.setObjectName("chooseRules")
        self.ruleName = QtWidgets.QLineEdit(parent=self.centralwidget)
        self.ruleName.setGeometry(QtCore.QRect(650, 220, 281, 20))
        self.ruleName.setReadOnly(True)
        self.ruleName.setObjectName("ruleName")
        self.openRules = QtWidgets.QPushButton(parent=self.centralwidget)
        self.openRules.setGeometry(QtCore.QRect(850, 250, 81, 23))
        self.openRules.setObjectName("openRules")
        self.label_16 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(100, 340, 47, 20))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.txtRule = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        self.txtRule.setGeometry(QtCore.QRect(93, 370, 791, 81))
        self.txtRule.setReadOnly(True)
        self.txtRule.setObjectName("txtRule")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 969, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.chooseRules.clicked.connect(self.chooseRulesFile)
        self.openRules.clicked.connect(self.openRulesFile)
        self.predictButt.clicked.connect(self.predictInputData)
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.predictButt.setText(_translate("MainWindow", "Predict"))
        self.sex.setItemText(0, _translate("MainWindow", "Female"))
        self.sex.setItemText(1, _translate("MainWindow", "Male"))
        self.age.setCurrentText(_translate("MainWindow", "Adult"))
        self.age.setItemText(0, _translate("MainWindow", "Adult"))
        self.age.setItemText(1, _translate("MainWindow", "MidleAge"))
        self.age.setItemText(2, _translate("MainWindow", "Old"))
        self.chestPain.setItemText(0, _translate("MainWindow", "0"))
        self.chestPain.setItemText(1, _translate("MainWindow", "1"))
        self.chestPain.setItemText(2, _translate("MainWindow", "2"))
        self.chestPain.setItemText(3, _translate("MainWindow", "3"))
        self.restRB.setItemText(0, _translate("MainWindow", "Low"))
        self.restRB.setItemText(1, _translate("MainWindow", "Normal"))
        self.restRB.setItemText(2, _translate("MainWindow", "High"))
        self.chol.setItemText(0, _translate("MainWindow", "Normal"))
        self.chol.setItemText(1, _translate("MainWindow", "High Risk"))
        self.chol.setItemText(2, _translate("MainWindow", "Extreme"))
        self.fastBS.setItemText(0, _translate("MainWindow", "=<120"))
        self.fastBS.setItemText(1, _translate("MainWindow", ">120"))
        self.resElec.setItemText(0, _translate("MainWindow", "Normal"))
        self.resElec.setItemText(1, _translate("MainWindow", "Abnormal ST-T Waves"))
        self.resElec.setItemText(2, _translate("MainWindow", "Left Ventricular Hypertrophy"))
        self.thalach.setItemText(0, _translate("MainWindow", "Low"))
        self.thalach.setItemText(1, _translate("MainWindow", "Normal"))
        self.thalach.setItemText(2, _translate("MainWindow", "High"))
        self.exang.setItemText(0, _translate("MainWindow", "No"))
        self.exang.setItemText(1, _translate("MainWindow", "Yes"))
        self.oldpeak.setItemText(0, _translate("MainWindow", "Low"))
        self.oldpeak.setItemText(1, _translate("MainWindow", "Normal"))
        self.oldpeak.setItemText(2, _translate("MainWindow", "High"))
        self.slope.setItemText(0, _translate("MainWindow", "Upward Slope"))
        self.slope.setItemText(1, _translate("MainWindow", "Flat"))
        self.slope.setItemText(2, _translate("MainWindow", "Downward Slope"))
        self.ca.setItemText(0, _translate("MainWindow", "0"))
        self.ca.setItemText(1, _translate("MainWindow", "1"))
        self.ca.setItemText(2, _translate("MainWindow", "2"))
        self.ca.setItemText(3, _translate("MainWindow", "3"))
        self.thal.setItemText(0, _translate("MainWindow", "Normal"))
        self.thal.setItemText(1, _translate("MainWindow", "Fixed Defect"))
        self.thal.setItemText(2, _translate("MainWindow", "Reversable Defect"))
        self.label.setText(_translate("MainWindow", "Age: "))
        self.label_2.setText(_translate("MainWindow", "Sex:"))
        self.label_3.setText(_translate("MainWindow", "Chest Pain Type:"))
        self.label_4.setText(_translate("MainWindow", "Resting Blood Presure:"))
        self.label_5.setText(_translate("MainWindow", "Serum Cholesterol:"))
        self.label_6.setText(_translate("MainWindow", "Fasting Blood Sugar:"))
        self.label_7.setText(_translate("MainWindow", "Resting Electrocardiographic Results:"))
        self.label_8.setText(_translate("MainWindow", "Maximum Heart Rate:"))
        self.label_9.setText(_translate("MainWindow", "Exercise Induced Anginan:"))
        self.label_10.setText(_translate("MainWindow", " ST depression:"))
        self.label_11.setText(_translate("MainWindow", "Slope of peak:"))
        self.label_12.setText(_translate("MainWindow", "Number of major vessels:"))
        self.label_13.setText(_translate("MainWindow", "Thalium Stress Type:"))
        self.label_14.setText(_translate("MainWindow", "Heart Disease Predict Application"))
        self.label_15.setText(_translate("MainWindow", "Result:"))
        self.chooseRules.setText(_translate("MainWindow", "Choose Rules"))
        self.openRules.setText(_translate("MainWindow", "Detail Rules"))
        self.label_16.setText(_translate("MainWindow", "Rule:"))
    def chooseRulesFile(self):
        filepath = QFileDialog.getOpenFileName()
        filename = re.search(r'[^/]+$', filepath[0]).group()
        self.ruleName.setText(filename)
    def openRulesFile(self):
        file_name = self.ruleName.text()
        if file_name == "":
            return
        subprocess.run(['start', file_name], check=True, shell=True)
    def predictInputData(self):
        self.sex.currentText()
        if self.sex.currentText() == 'Male':
            sex = 0
        else:
            sex = 1
        if self.fastBS.currentText() == '=<120':
            fast = 0
        else:
            fast = 1
        if self.resElec.currentText() == 'Normal':
            res = 0
        elif self.resElec.currentText() == 'Abnormal ST-T Waves':
            res = 1
        elif self.resElec.currentText() == 'Left Ventricular Hypertrophy':
            res = 2
        if self.exang.currentText() == 'No':
            exang = 0
        else:
            exang = 1
        if self.slope.currentText() == 'Upward Slope':
            slope = 0
        elif self.slope.currentText() == 'Flat':
            slope = 1
        elif self.slope.currentText() == 'Downward Slope':
            slope = 2
        if self.thal.currentText() == 'Normal':
            thal = 1
        elif self.thal.currentText() == 'Fixed Defect':
            thal = 2
        elif self.thal.currentText() == 'Reversable Defect':
            thal = 3
        instance =     {'Age': self.age.currentText(), 'sex': sex, 'cp': int(self.chestPain.currentText()),
                       'trest': self.restRB.currentText(), 'chol': self.chol.currentText(),
                       'fbs': fast, 'restecg': res,
                       'thalach': self.thalach.currentText(), 'exang': exang,
                       'oldpeak': self.oldpeak.currentText(), 'slope': slope,
                       'ca': int(self.ca.currentText()), 'thal': thal
                       }
        print(instance)
        pathname = self.ruleName.text()
        if pathname == "":
            return
        prediction = predict_disease(instance, pathname)
        self.txtRule.setPlainText(str(prediction))
        if prediction['Prediction']==0:
            self.lineEdit.setText('Not Sick')
        elif prediction['Prediction']==1:
            self.lineEdit.setText('Sick')
        else:
            self.lineEdit.setText('Unknown')

# Hàm đọc tập luật từ file CSV
def load_rules_from_csv(file_path):
    return pd.read_csv(file_path).to_dict(orient='records')
# Hàm dự đoán từ tập luật
def predict_from_rules(rules, instance):
    for rule in rules:
        match = True
        for key, value in rule.items():
            if key != 'Prediction':
                if key not in instance:
                    match = False
                    break
                # So sánh giá trị thuộc tính với đối tượng mới
                if pd.isna(value):
                    continue
                elif instance[key] != value:
                    match = False
                    break
        if match:
            return rule
    return 'Unknown'
# Hàm kiểm tra đối tượng mới và dự đoán bệnh
def predict_disease(new_instance, pathname):
    # Đọc tập luật từ file CSV
    loaded_rules = load_rules_from_csv(pathname)

    # Dự đoán từ tập luật
    prediction = predict_from_rules(loaded_rules, new_instance)
    return prediction
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())

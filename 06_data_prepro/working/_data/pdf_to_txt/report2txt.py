#한글 pdf 변환 후 pdf txt파일 변환  

import re
from pathlib import Path
import hanja
import pandas as pd
import os
import shutil
import win32com.client as win32
import pdfplumber
# set arguments
# path will be set as this directory if it's not defined  



#set folder path
input_dir = os.path.join(os.getcwd())
output_dir = os.path.join(input_dir, 'text')
new_pdf = os.path.join(input_dir, 'new_pdf')

# change ' ' to '_' in file name
def changeFileName():
    for (root,_, files) in os.walk(input_dir):
        for file in files:
            if file[-4:] == '.pdf' or file[-4:] == '.hwp':
                new_file = file.replace(" ","_")
                os.rename(os.path.join(root, file), os.path.join(root, new_file))



# collect path and file_list of .pdf file
def allfile(input_dir):
    file_list = []
    path_list = []
    for (root, directories, files) in os.walk(input_dir):
        for file in files:
            if file[-4:] == '.pdf' or file[-4:] == '.hwp':
                file_path = os.path.join(root, file)
                path_list.append(file_path)
                ori_file_path = file_path.split('\\')[-1]
                file_list.append(ori_file_path)
    return path_list, file_list

	
#convert hwp to pdf
def hwp2pdf(file_path_list,file_name_list):
    hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")  # 한글프로그램 실행
    hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")  # 보안모듈 적용(파일 열고닫을 때 팝업이 안나타남)
    hwp.XHwpWindows.Item(0).Visible = True  # 한글 백그라운드 한글 보이게
    for i,path in enumerate(file_path_list):
        if path[-4:] == '.hwp':
            new_path = os.path.join(new_pdf,file_name_list[i][:-4]+".pdf")
            hwp.Open(path)
            hwp.HAction.GetDefault("FileSaveAsPdf", hwp.HParameterSet.HFileOpenSave.HSet)  # 파일 저장 액션의 파라미터를
            hwp.HParameterSet.HFileOpenSave.filename = f'{new_path}' # 저장할 파일 경로 및 이름.pdf 확장자명을 꼭 pdf로 적어주어야 함.
            hwp.HParameterSet.HFileOpenSave.Format = "PDF" # 파일 확장자 pdf
            hwp.HParameterSet.HFileOpenSave.Attributes = 16384
            hwp.HAction.Execute("FileSaveAsPdf", hwp.HParameterSet.HFileOpenSave.HSet)


# convert .pdf to .txt
def text(file_path_list,file_name_list):

    for i,path in enumerate(file_path_list):
        if path[-4:] == '.pdf':     # 파일명이 .pdf 로 끝날 시 
            pdf = pdfplumber.open(path)
            pages = len(pdf.pages)
            name = file_name_list[i][:-4] + ".txt"  # 
            f = open(output_dir+'\\'+name,'w',encoding='UTF-8')
            for k in range(pages):              # 변환된 txt 파일중 특수문자, 줄 바꿈 부분을 삭제한다. 
                page = pdf.pages[k].extract_text()
                page = hanja.translate(page, 'substitution') 
                page = re.sub('[^A-Za-z0-9가-힣\s]', '', page)
                page = page.replace('\n','')
                f.write(page)                                  
            f.close()
        else:
            continue




if __name__ == "__main__":

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(new_pdf):
        os.mkdir(new_pdf)
        

    changeFileName()
    file_path_list, file_name_list = allfile(input_dir)


    hwp2pdf(file_path_list,file_name_list)
    file_path_list, file_name_list = allfile(input_dir)
    text(file_path_list,file_name_list)
    print("전체 파일개수 : ", len(os.listdir(output_dir)))

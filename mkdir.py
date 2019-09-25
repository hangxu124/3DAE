import os
import cv2
import numpy as np
'''
def mk(mode):
    if mode == "a":
        an = ["ar","dr","ml","mr","rb","tp","tl","tr"]
        dad = "/usr/home/sut/v_AD"
        for i in range(13,15):
            for j in an:
                for k in range(30):
                    folder = os.path.join(dad,str(i)+"-"+j+"-"+"fro"+"-"+str(k))
                    os.mkdir(folder)
    if mode == "n":
        no = ["n1","n2","n3","n4","n5","n6"]
        nad = "/usr/home/sut/V_AD"
        for i in range(13,15):
            for j in no:
                for k in range(223):
                    folder = os.path.join(nad,str(i)+"-"+j+"-"+"fro"+"-"+str(k))
                    os.mkdir(folder)

def cat(mode):
    if mode == "a":
        an1=["adjusting_radio","drinking","messaging_left","messaging_right","reaching_behind","talking_with_passenger",
        "talking_with_phone_left","talking_with_phone_right"]
        an2 = ["ar","dr","ml","mr","rb","tp","tl","tr"]
        hpath = "/usr/home/sut/Datasets/DAD"
        dad = "/usr/home/sut/v_AD"
        for i in range(13,15):
            t = "Tester"+str(i)
            for a1,a2 in zip(an1,an2):
                ir_path = os.path.join(hpath,t,a1,"front_IR")
                depth_path = os.path.join(hpath,t,a1,"front_depth")
                #ir_img = sorted(os.listdir(ir_path))
                #depth_img = sorted(os.listdir(depth_path))
                for n in range(1350):
                    imgname = "img_"+str(n)+".png"
                    ir_img = cv2.imread(os.path.join(ir_path,imgname),0)
                    depth_img = cv2.imread(os.path.join(depth_path,imgname),0)
                    save_img = np.stack((ir_img,depth_img),axis=0)
                    save_path = os.path.join(dad, str(i)+"-"+a2+"-"+"fro"+"-"+str(int(n/45)),'{:02d}.npy'.format(((n)%45)+1))
                    np.save(save_path,save_img)
    if mode == "n":
        an1=["normal_driving_1","normal_driving_2","normal_driving_3","normal_driving_4","normal_driving_5","normal_driving_6"]
        an2 = ["n1","n2","n3","n4","n5","n6"]
        hpath = "/usr/home/sut/Datasets/DAD"
        dad = "/usr/home/sut/V_AD"
        for i in range(13,15):
            t = "Tester"+str(i)
            for a1,a2 in zip(an1,an2):
                ir_path = os.path.join(hpath,t,a1,"front_IR")
                depth_path = os.path.join(hpath,t,a1,"front_depth")
                #ir_img = sorted(os.listdir(ir_path))
                #depth_img = sorted(os.listdir(depth_path))
                for n in range(10000):
                    imgname = "img_"+str(n)+".png"
                    ir_img = cv2.imread(os.path.join(ir_path,imgname),0)
                    depth_img = cv2.imread(os.path.join(depth_path,imgname),0)
                    save_img = np.stack((ir_img,depth_img),axis=0)
                    save_path = os.path.join(dad, str(i)+"-"+a2+"-"+"fro"+"-"+str(int(n/45)),'{:02d}.npy'.format((n%45)+1))
                    np.save(save_path,save_img)

'''
def mk():
    path = "/usr/home/sut/DAD"
    for i in range(22,26):
        s = "Tester"+ '{:02d}'.format(i)
        #os.mkdir(os.path.join(path,s))
    tester = sorted(os.listdir(path))
    print (tester)
    motion = ["adjusting_radio","drinking","messaging_left","messaging_right","reaching_behind","talking_with_passenger",
        "talking_with_phone_left","talking_with_phone_right",
        "normal_driving_1","normal_driving_2","normal_driving_3","normal_driving_4","normal_driving_5","normal_driving_6"]
    view =["front","top"]
    camera = ["IR","depth"]
    for t in tester:
        if t in ["Tester22","Tester24","Tester23","Tester25"]:
            for m in motion:
                os.mkdir(os.path.join(path,t,m))
                if "normal" in m:
                    for i in range(1,223):
                        os.mkdir(os.path.join(path,t,m,"{:03d}".format(i)))
                        for v in view:
                            os.mkdir(os.path.join(path,t,m,"{:03d}".format(i),v))
                            for c in camera:
                                os.mkdir(os.path.join(path,t,m,"{:03d}".format(i),v,c))
                else:
                    for i in range(1,31):
                        os.mkdir(os.path.join(path,t,m,("{:03d}".format(i))))
                        for v in view:
                            os.mkdir(os.path.join(path,t,m,"{:03d}".format(i),v))
                            for c in camera:
                                os.mkdir(os.path.join(path,t,m,"{:03d}".format(i),v,c))
    
import shutil
def move():
    source = "/usr/home/sut/Datasets/DAD"
    target = "/usr/home/sut/DAD"
    # testers = os.listdir(source)
    testers = ["Tester22","Tester23","Tester24"]
    for tester in testers:
        to_tester = "Tester"+"{:02d}".format(int(tester[6:]))
        for motion in os.listdir(os.path.join(source,tester)):
            to_motion = motion
            for camera_view in os.listdir(os.path.join(source,tester,motion)):
                to_camera = camera_view.split("_")[1]
                to_view = camera_view.split("_")[0]
                if to_view == "upper":
                    to_view ="top"
                imgs = os.listdir(os.path.join(source,tester,motion,camera_view))
                for img in imgs:
                    to_folder = "{:03d}".format(int(int(img.split(".")[0][4:])/45)+1)
                    to_number = "{:02d}".format(int(int(img.split(".")[0][4:])%45)+1)+".png"
                    if to_folder<="222":
                        old_path = os.path.join(source,tester,motion,camera_view,img)
                        new_path = os.path.join(target,to_tester,to_motion,to_folder,to_view,to_camera,to_number)
                        shutil.copyfile(old_path,new_path)

def move_vali():
    source = "/usr/home/sut/Datasets/DAD/val06"
    target = "/usr/home/sut/DAD/val06"
    for motion in sorted(os.listdir(source)):
        to_motion = motion
        for camera_view in os.listdir(os.path.join(source,motion)):
            to_camera = camera_view.split("_")[1]
            to_view = camera_view.split("_")[0]
            imgs = os.listdir(os.path.join(source,motion,camera_view))
            for img in imgs:
                to_folder = "{:03d}".format(int(int(img.split(".")[0][4:])/45)+1)
                to_number = "{:02d}".format(int(int(img.split(".")[0][4:])%45)+1)+".png"
                if to_folder<="222":
                    old_path = os.path.join(source,motion,camera_view,img)
                    new_path = os.path.join(target,to_motion,to_folder,to_view,to_camera,to_number)
                    shutil.copyfile(old_path,new_path)



def gen_train_list():
    folders = ["Tester22","Tester23","Tester24"]
    source = "/usr/home/sut/DAD"
    for folder in folders:
        for motion in os.listdir(os.path.join(source,folder)):
            for num in os.listdir(os.path.join(source,folder,motion)):
                flag =1 if "normal" in motion else 2
                with open("tl.txt","a") as file:
                    file.write(os.path.join(folder,motion,num)+" "+str(flag)+"\n")




def val():
    source = "/usr/home/sut/Datasets/DAD/val06/normal_driving_1"
    target = "/usr/home/sut/DAD/val006/normal_driving_1"
    v6n1= np.load("/usr/home/sut/label/v6n1.npy")
    # for i in range(1,9955):
    #     f1 = os.path.join(target,"{:04d}".format(i))
    #     os.mkdir(f1)
    #     os.mkdir(os.path.join(f1,"top"))
    #     os.mkdir(os.path.join(f1,"front"))
    #     os.mkdir(os.path.join(f1,"top","depth"))
    #     os.mkdir(os.path.join(f1,"top","IR"))
    #     os.mkdir(os.path.join(f1,"front","depth"))
    #     os.mkdir(os.path.join(f1,"front","IR"))
    folders = os.listdir(source)
    for folder in folders:
        camera, di = folder.split("_")
        from_f = os.path.join(source,folder)
        for i in range(1,9955,10):
            print (i)
            to_f = os.path.join(target,"{:04d}".format(i),camera, di)
            if not os.path.exists(to_f):
                os.makedirs(to_f)
            num = 1      
            for i in range(i-1, i+44):
                old_path = os.path.join(from_f,"img_"+str(i)+".png")
                new_path = os.path.join(to_f,"{:02d}".format(num)+".png")
                num +=1
                shutil.copyfile(old_path,new_path)

            num=1
    for i in range(1,9955,10):
        with open("sliding.txt","a") as sw:
                sw.write(os.path.join("val006","normal_driving_1","{:04d}".format(i))+" "+str(int(v6n1[:,i+22]+1))+"\n")

def make_val_():
    source = "/usr/home/sut/DAD/"

    folders = sorted( os.listdir(source))
    for f in folders:
        with open("vallist_01.txt","a") as file:
            file.write(os.path.join(source,f)+" "+"1"+"\n")



def make_train_val():
    path = "/usr/home/sut/DAD/val06"
    normals =sorted( os.listdir(path))
    v2n1= np.load("/usr/home/sut/label/v2n1.npy")
    v2n2= np.load("/usr/home/sut/label/v2n2.npy")
    v2n3= np.load("/usr/home/sut/label/v2n3.npy")
    v2n4= np.load("/usr/home/sut/label/v2n4.npy")
    v2n5= np.load("/usr/home/sut/label/v2n5.npy")
    v2n6= np.load("/usr/home/sut/label/v2n6.npy")
    
    v4n1= np.load("/usr/home/sut/label/v4n1.npy")
    v4n2= np.load("/usr/home/sut/label/v4n2.npy")
    v4n3= np.load("/usr/home/sut/label/v4n3.npy")
    v4n4= np.load("/usr/home/sut/label/v4n4.npy")
    v4n5= np.load("/usr/home/sut/label/v4n5.npy")
    v4n6= np.load("/usr/home/sut/label/v4n6.npy")

    v1n1= np.load("/usr/home/sut/label/v1n1.npy")
    v1n2= np.load("/usr/home/sut/label/v1n2.npy")
    v1n3= np.load("/usr/home/sut/label/v1n3.npy")
    v1n4= np.load("/usr/home/sut/label/v1n4.npy")
    v1n5= np.load("/usr/home/sut/label/v1n5.npy")
    v1n6= np.load("/usr/home/sut/label/v1n6.npy")

    v5n1= np.load("/usr/home/sut/label/v5n1.npy")
    v5n2= np.load("/usr/home/sut/label/v5n2.npy")
    v5n3= np.load("/usr/home/sut/label/v5n3.npy")
    v5n4= np.load("/usr/home/sut/label/v5n4.npy")
    v5n5= np.load("/usr/home/sut/label/v5n5.npy")
    v5n6= np.load("/usr/home/sut/label/v5n6.npy")


    v6n1= np.load("/usr/home/sut/label/v6n1.npy")
    v6n2= np.load("/usr/home/sut/label/v6n2.npy")
    v6n3= np.load("/usr/home/sut/label/v6n3.npy")
    v6n4= np.load("/usr/home/sut/label/v6n4.npy")
    v6n5= np.load("/usr/home/sut/label/v6n5.npy")
    v6n6= np.load("/usr/home/sut/label/v6n6.npy")
    #print (np.shape(v2n1))
    # print (normals)
    for n in normals:
        if "6" in n:
            numbers = sorted(os.listdir(os.path.join(path,n)))
            for number in numbers:
                s = os.path.join("val05",n, number)
                l = str(int(v5n6[:,(int(number)*45 +23-45)])+1)

                with open("val5.txt","a") as file:
                    file.write(s+" "+l+"\n")


    #for tester in testers:
    #    if tester == 'Tester20' or tester == 'Tester21':
            # motions = os.listdir(os.path.join(path,tester))
            # for motion in motions:
                # if  ( "normal" in motion):
                    # numbers = os.listdir (os.path.join(path,tester,motion))
                    # for number in numbers:
                        # s = os.path.join(tester,motion,number)
                        # with open("trainlist_1.txt","a") as file:
                            # file.write(s+" "+"1"+"\n")
    

def copy_a():
    path = "/usr/home/sut/DAD"
    testers = os.listdir(path)
    for tester in testers:
        motions = os.listdir(os.path.join(path,tester))
        for motion in motions:
            if not ("normal" in motion):
                numbers = os.listdir(os.path.join(path,tester,motion))
                for number in numbers:
                    ni = int(number)
                    n1 = "{:03d}".format(ni+30)
                    n2 = "{:03d}".format(ni+60)
                    n3 = "{:03d}".format(ni+90)
                    n4 = "{:03d}".format(ni+120)
                    n5 = "{:03d}".format(ni+150)
                    shutil.copytree(os.path.join(path,tester,motion,number),
                    os.path.join(path,tester,motion,n1))
                    shutil.copytree(os.path.join(path,tester,motion,number),
                    os.path.join(path,tester,motion,n2))
                    shutil.copytree(os.path.join(path,tester,motion,number),
                    os.path.join(path,tester,motion,n3))
                    shutil.copytree(os.path.join(path,tester,motion,number),
                    os.path.join(path,tester,motion,n4))
                    shutil.copytree(os.path.join(path,tester,motion,number),
                    os.path.join(path,tester,motion,n5))
                    




def val_shuffle():
    path = "/usr/home/sut/3D-ResNets-PyTorch/vallist.txt"
    with open (path,"r") as f:
        lines = f.readlines()
    import random
    random.shuffle(lines)
    for i in lines:
        with open("vallist01.txt","a") as a:
            a.write(i)


def _val():
    fp = "/usr/home/sut/Datasets/DAD"
    val =["val01","val05"]
    tp = "/usr/home/sut/DAD"
    motion = ["normal_driving_1","normal_driving_2","normal_driving_3","normal_driving_4","normal_driving_5","normal_driving_6"]
    view = ["front","top"]
    camera = ["depth","IR"]
    for folder in val:
        for m in motion:
            #os.mkdir(os.path.join(tp,folder,m))
            for i in range(1,223):
                for v in view:
                    for c in camera:
                        os.mkdir(os.path.join(tp,folder,m, "{:03d}".format(i),v,c))



def extract_label():
    label = "/usr/home/sut/MyRes3D_AE/vallist01.txt"
    newlabel = "/usr/home/sut/MyRes3D_AE/vallabel.txt"
    with open (label,"r") as lb:
        lines = lb.readlines()
    for line in lines:
        n = str(int(line.split()[1])-1)
        with open(newlabel,"a") as new:
            new.write(n+"\n")


if __name__ == "__main__":
    val()
    #mk()
    # gen_train_list()
    # move()
    # make_train_val()
    #make_val_()
    #val_shuffle()
    #copy_a()
    # move_vali()
    # extract_label()
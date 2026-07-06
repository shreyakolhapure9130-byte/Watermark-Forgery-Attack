"""Reproduce the base 200-image submission (score 0.624):
WM_1 DwtDct, WM_2 RivaGAN, WM_7 TrustMark re-embedded natively; WM_4/5/6 copy attack.
Requires: pip install invisible-watermark trustmark opencv-python numpy scipy pillow
Dataset at ./Dataset ."""
import numpy as np, cv2, zipfile
from pathlib import Path
from collections import Counter
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
from imwatermark import WatermarkEncoder, WatermarkDecoder
from trustmark import TrustMark

DATASET=Path("Dataset"); OUT=Path("submission_temp"); OUT.mkdir(exist_ok=True)
GLOBAL_SCALE=2.0
CATS=[("WM_1",1,25),("WM_2",26,50),("WM_3",51,75),("WM_4",76,100),
      ("WM_5",101,125),("WM_6",126,150),("WM_7",151,175),("WM_8",176,200)]
# copy-attack extractor per additive group (band chosen by max d'):
EXTRACT={"WM_4":("ghp",0.5),"WM_5":("medhp",3),"WM_6":("medhp",5)}

WatermarkEncoder.loadModel(); WatermarkDecoder.loadModel()
tm=TrustMark(verbose=False, model_type="Q")
def bgr(p): return cv2.imread(str(p),cv2.IMREAD_COLOR)
def rgb(p): return np.array(cv2.cvtColor(bgr(p),cv2.COLOR_BGR2RGB),dtype=np.float32)
def parse(r): return (next((x for x in r if isinstance(x,bool)),None),
                      next((x for x in r if isinstance(x,str)),None))
def srcs(g): return sorted((DATASET/"watermarked_sources"/g).glob("*.png"))
def vote(g,m,L):
    d=WatermarkDecoder("bits",L)
    A=np.stack([np.asarray(d.decode(bgr(p),m),dtype=int) for p in srcs(g)])
    return (A.mean(0)>0.5).astype(int)
def extract(imgs,spec):
    k,s=spec
    r=[im-(gaussian_filter(im,(s,s,0)) if k=="ghp" else median_filter(im,size=(int(s),int(s),1))) for im in imgs]
    d=np.median(r,0); return d-d.mean()
def gain(delta,S,T):
    v=delta.ravel(); nv=np.linalg.norm(v)+1e-9; proj=lambda im:(im.ravel()-im.mean())@v/nv
    return (np.mean([proj(im) for im in S])-np.mean([proj(im) for im in T]))/nv

msg1=vote("WM_1","dwtDct",48); msg2=vote("WM_2","rivaGan",32)
sec7=Counter([parse(tm.decode(Image.open(p).convert("RGB")))[1] for p in srcs("WM_7")
              if parse(tm.decode(Image.open(p).convert("RGB")))[1]]).most_common(1)[0][0]

for g,a,b in CATS:
    tps=[DATASET/"clean_targets"/f"{n}.png" for n in range(a,b+1)]
    if g=="WM_1":
        e=WatermarkEncoder(); e.set_watermark("bits",list(map(int,msg1)))
        for p in tps: cv2.imwrite(str(OUT/p.name), e.encode(bgr(p),"dwtDct"))
    elif g=="WM_2":
        e=WatermarkEncoder(); e.set_watermark("bits",list(map(int,msg2)))
        for p in tps: cv2.imwrite(str(OUT/p.name), e.encode(bgr(p),"rivaGan"))
    elif g=="WM_7":
        for p in tps:
            c=Image.open(p).convert("RGB"); st=tm.encode(c,sec7)
            if st.size!=c.size: st=st.resize(c.size)
            st.save(OUT/p.name)
    elif g in EXTRACT:
        S=[rgb(p) for p in srcs(g)]; T=[rgb(p) for p in tps]
        d=extract(S,EXTRACT[g]); gg=gain(d,S,T)*GLOBAL_SCALE
        for p,t in zip(tps,T): Image.fromarray(np.clip(t+gg*d,0,255).astype(np.uint8)).save(OUT/p.name)
    else:  # WM_3, WM_8 handled by wmcopier/ ; copy clean here as placeholder
        for p in tps: cv2.imwrite(str(OUT/p.name), bgr(p))
    print(g,"done")

with zipfile.ZipFile("submission.zip","w",zipfile.ZIP_DEFLATED) as z:
    for im in OUT.glob("*.png"): z.write(im,arcname=im.name)
print("wrote submission.zip")

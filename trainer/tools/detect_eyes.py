"""Offline YOLO eye annotations; source and training-bucket coordinate spaces."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def detections(result):
    rows=[]
    polygons=result.masks.xy if result.masks is not None else []
    for i,box in enumerate(result.boxes):
        rows.append(dict(xyxy=box.xyxy[0].cpu().tolist(),
                         confidence=float(box.conf[0]),class_id=int(box.cls[0]),
                         polygon=polygons[i].tolist() if i<len(polygons) else []))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('config');ap.add_argument('--model',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--device',default='cpu')
    ap.add_argument('--conf',type=float,default=.25)
    ap.add_argument('--imgsz',type=int,default=1024)
    args=ap.parse_args()
    from ultralytics import YOLO
    import ultralytics
    from trainer.training.config import load_config
    from trainer.data.cache import load_and_crop
    cfg=load_config(args.config);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for folder in ['annotations','masks','previews']:(out/folder).mkdir(exist_ok=True)
    model=YOLO(args.model)
    if model.task!='segment':raise ValueError('Expected an eye segmentation model')
    files=sorted({p.resolve() for subset in cfg.dataset.effective_subsets()
                  for p in Path(subset.path).rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp'}})
    with open(args.model,'rb') as f:model_hash=hashlib.file_digest(f,'sha256').hexdigest()
    manifest=dict(model=str(Path(args.model).resolve()),model_sha256=model_hash,
                  ultralytics_version=ultralytics.__version__,classes=model.names,
                  confidence=args.conf,imgsz=args.imgsz,images=[])
    counts=Counter();review=[];thumbs=[]
    for index,path in enumerate(files):
        with Image.open(path) as im:original=im.convert('RGB')
        original_result=model.predict(original,imgsz=args.imgsz,conf=args.conf,iou=.5,
                                      device=args.device,retina_masks=True,verbose=False)[0]
        original_rows=detections(original_result)
        record=dict(source=str(path),source_size=list(original.size),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    original=original_rows,training=[])
        identity=hashlib.sha256(str(path).encode()).hexdigest()[:16]
        for resolution in cfg.dataset.tiers:
            prepared,meta=load_and_crop(path,cfg.dataset.build_bucket_manager(resolution))
            result=model.predict(prepared,imgsz=args.imgsz,conf=args.conf,iou=.5,
                                 device=args.device,retina_masks=True,verbose=False)[0]
            eyes=detections(result);counts[len(eyes)]+=1
            stem=f'{path.stem}-{identity}-{resolution}'
            mask=Image.new('L',prepared.size);md=ImageDraw.Draw(mask)
            for eye in eyes:
                if len(eye['polygon'])>=3:md.polygon([tuple(p) for p in eye['polygon']],fill=255)
            mask.save(out/'masks'/f'{stem}.png')
            entry=dict(resolution=resolution,size=list(prepared.size),metadata=meta,
                       eyes=eyes,mask=f'masks/{stem}.png')
            record['training'].append(entry)
            # Diagnostic overlays are separate artifacts; source images are unchanged.
            preview=prepared.copy();draw=ImageDraw.Draw(preview)
            for eye in eyes:
                draw.rectangle(eye['xyxy'],outline='lime',width=3)
                draw.text((eye['xyxy'][0],max(0,eye['xyxy'][1]-14)),f"{eye['confidence']:.2f}",fill='lime')
                if len(eye['polygon'])>=3:
                    points=[tuple(p) for p in eye['polygon']];draw.line(points+[points[0]],fill='cyan',width=2)
            preview.thumbnail((768,768));preview.save(out/'previews'/f'{stem}.jpg',quality=92)
            entry['preview']=f'previews/{stem}.jpg'
            if len(eyes)==0 or len(eyes)>2 or any(e['confidence']<.5 for e in eyes) or len(eyes)!=len(original_rows):
                review.append(dict(source=str(path),resolution=resolution,count=len(eyes),original_count=len(original_rows),preview=entry['preview']))
            if index%max(1,len(files)//20)==0:
                tile=Image.new('RGB',(256,280),'#222222');small=ImageOps.contain(preview,(256,256));tile.paste(small,((256-small.width)//2,0));ImageDraw.Draw(tile).text((4,260),f'{index}: {len(eyes)} eyes',fill='white');thumbs.append(tile)
        ann=f'annotations/{path.stem}-{identity}.json'
        (out/ann).write_text(json.dumps(record,indent=2)+'\n')
        manifest['images'].append(dict(source=str(path),annotation=ann))
        if (index+1)%20==0:print(f'{index+1}/{len(files)}',flush=True)
    manifest['training_detection_counts']=dict(counts);manifest['review']=review
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if thumbs:
        sheet=Image.new('RGB',(256*4,280*((len(thumbs)+3)//4)),'#222222')
        for i,tile in enumerate(thumbs):sheet.paste(tile,((i%4)*256,(i//4)*280))
        sheet.save(out/'contact-sheet.jpg',quality=95)
    print(json.dumps(dict(images=len(files),training_detection_counts=dict(counts),review=len(review)),indent=2))

if __name__=='__main__':main()

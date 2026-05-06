import { invoke }             from '@tauri-apps/api/core';
import { listen }             from '@tauri-apps/api/event';
import { save as saveDialog } from '@tauri-apps/plugin-dialog';

// ─── Types ────────────────────────────────────────────────────────────────────

type Tool      = 'select' | 'atom' | 'bond' | 'erase';
type BondOrder = 1 | 2 | 3 | 'a';
type RightTab  = 'props' | 'qm' | 'dock' | 'export';

interface LbAtom {
  id:      number;
  el:      string;
  x:       number;
  y:       number;
  z:       number;
  charge:  number;
  radical: number;
}

interface LbBond {
  id:       number;
  a1:       number;
  a2:       number;
  order:    number;
  aromatic: boolean;
}

interface Pos2D { x: number; y: number; }

// Payload emitted by lb:min-progress events
interface MinProgressPayload {
  step:    number;
  total:   number;   // 0 = indeterminate (external process)
  energy:  number;
  method:  string;
  message: string;
}

// Payload emitted by lb:min-done events
interface MinDonePayload {
  coords:       { x: number; y: number; z: number }[];
  steps_run:    number;
  final_energy: number;
  method:       string;
}

// ─── Docking types ────────────────────────────────────────────────────────────

interface DockProgressPayload {
  message: string;
}

interface DockPose {
  rank:     number;
  affinity: number;   // kcal/mol
  rmsd_lb:  number;
  rmsd_ub:  number;
  coords:   { x: number; y: number; z: number }[];
}

interface DockDonePayload {
  poses:         DockPose[];
  best_affinity: number;
  engine:        string;
  output_pdbqt:  string;  // full path to the output PDBQT file in temp dir (for save-all)
}

interface ElementDatum { color: string; r: number; valence: number; mw: number; }
interface RingTemplate  { n: number; elems: string[]; bonds: number[]; aromatic?: boolean; }

// ─── Constants ────────────────────────────────────────────────────────────────

const BL3 = 1.54;
const BL2 = 40;
const DRAG_THRESHOLD = 4;

const ELEMENT_DATA: Record<string, ElementDatum> = {
  H:  { color: '#c8c8cc', r: 0.31, valence: 1, mw: 1.008   },
  C:  { color: '#888890', r: 0.76, valence: 4, mw: 12.011  },
  N:  { color: '#4488ff', r: 0.71, valence: 3, mw: 14.007  },
  O:  { color: '#ff4444', r: 0.66, valence: 2, mw: 15.999  },
  S:  { color: '#ddcc22', r: 1.05, valence: 2, mw: 32.065  },
  P:  { color: '#ffaa22', r: 1.07, valence: 3, mw: 30.974  },
  F:  { color: '#44ee44', r: 0.57, valence: 1, mw: 18.998  },
  Cl: { color: '#33cc33', r: 1.02, valence: 1, mw: 35.453  },
  Br: { color: '#aa2222', r: 1.20, valence: 1, mw: 79.904  },
  I:  { color: '#8844bb', r: 1.39, valence: 1, mw: 126.904 },
  B:  { color: '#ddaa55', r: 0.84, valence: 3, mw: 10.811  },
  Si: { color: '#bbbbaa', r: 1.11, valence: 4, mw: 28.086  },
};

const ORGANIC_ELEMS = ['C','N','O','S','P','H','F','Cl','Br','I','B','Si'] as const;

const MW_TABLE: Record<string, number> = {
  H:1.008,C:12.011,N:14.007,O:15.999,S:32.065,P:30.974,
  F:18.998,Cl:35.453,Br:79.904,I:126.904,B:10.811,Si:28.086,
};

const RINGS: Record<string, RingTemplate> = {
  cycloprop: { n:3, elems:['C','C','C'],              bonds:[1,1,1]              },
  cyclobut:  { n:4, elems:['C','C','C','C'],          bonds:[1,1,1,1]            },
  cyclopent: { n:5, elems:['C','C','C','C','C'],      bonds:[1,1,1,1,1]          },
  cyclohex:  { n:6, elems:['C','C','C','C','C','C'],  bonds:[1,1,1,1,1,1]        },
  benzene:   { n:6, elems:['C','C','C','C','C','C'],  bonds:[1,2,1,2,1,2], aromatic:true },
  pyrrole:   { n:5, elems:['N','C','C','C','C'],      bonds:[1,2,1,2,1],   aromatic:true },
  pyridine:  { n:6, elems:['N','C','C','C','C','C'],  bonds:[2,1,2,1,2,1], aromatic:true },
  imidazole: { n:5, elems:['N','C','N','C','C'],      bonds:[1,2,1,2,1],   aromatic:true },
  furan:     { n:5, elems:['O','C','C','C','C'],      bonds:[1,2,1,2,1],   aromatic:true },
  thiophene: { n:5, elems:['S','C','C','C','C'],      bonds:[1,2,1,2,1],   aromatic:true },
};

const FRAGS: Record<string, string> = {
  methyl:'C', ethyl:'CC', phenyl:'c1ccccc1',
  carboxyl:'C(=O)O', amino:'N', hydroxyl:'O',
};

// ─── Molecule state ───────────────────────────────────────────────────────────

let atoms:      LbAtom[] = [];
let bonds:      LbBond[] = [];
let nextAtomId: number   = 1;
let nextBondId: number   = 1;
let selAtomId:  number | null = null;
let tool:       Tool         = 'select';
let bondOrder:  BondOrder    = 1;
let currentEl:  string       = 'C';

const undoStack: string[] = [];
const redoStack: string[] = [];
const MAX_UNDO = 40;

// ─── 3D view state ────────────────────────────────────────────────────────────

let viewRot:  number[] = [1,0,0, 0,1,0, 0,0,1];
let viewZoom: number   = 60;
let viewPanX: number   = 0;
let viewPanY: number   = 0;

type DragMode = 'none' | 'rotate' | 'pan' | 'bond';
let ptrDown:      boolean                    = false;
let ptrStart:     { x: number; y: number }   = { x:0, y:0 };
let ptrLast:      { x: number; y: number }   = { x:0, y:0 };
let ptrMoved:     number                     = 0;
let dragMode:     DragMode                   = 'none';
// @ts-ignore
let _viewRotStart: number[]                   = [1,0,0, 0,1,0, 0,0,1];
let viewPanStart: { x: number; y: number }   = { x:0, y:0 };
let bondDragStart: number | null             = null;
let bondHoverAtom: number | null             = null;
let bondPreviewPx: { x: number; y: number } | null = null;

// ─── Minimization UI state ────────────────────────────────────────────────────

let minimizeRunning = false;

// ─── Docking state ────────────────────────────────────────────────────────────

let dockRunning      = false;
let dockPoses:       DockPose[]     = [];
let dockSelectedPose = 0;           // 0-based index into dockPoses
let dockOutputPdbqt  = '';          // path to output PDBQT for save-all
let dockReceptorPath = '';          // path used for last dock run (for visualizer windows)

// ─── DOM refs ─────────────────────────────────────────────────────────────────

let canvas:   HTMLCanvasElement;
let ctx:      CanvasRenderingContext2D;
let depSvg:   SVGSVGElement;
let depBonds: SVGGElement;
let depAtoms: SVGGElement;
let tooltip:  HTMLDivElement;

// ─── 3D Math ──────────────────────────────────────────────────────────────────

function rotVec(m:number[],x:number,y:number,z:number):[number,number,number] {
  return [m[0]*x+m[1]*y+m[2]*z, m[3]*x+m[4]*y+m[5]*z, m[6]*x+m[7]*y+m[8]*z];
}
function mat3Mul(A:number[],B:number[]): number[] {
  const C=new Array(9).fill(0);
  for(let i=0;i<3;i++) for(let k=0;k<3;k++) for(let j=0;j<3;j++) C[i*3+j]+=A[i*3+k]*B[k*3+j];
  return C;
}
function mat3Transpose(m:number[]): number[] {
  return [m[0],m[3],m[6], m[1],m[4],m[7], m[2],m[5],m[8]];
}
function project(a:LbAtom): { sx:number; sy:number; sz:number } {
  const [vx,vy,vz]=rotVec(viewRot,a.x,a.y,a.z);
  return { sx:vx*viewZoom+canvas.width/2+viewPanX, sy:-vy*viewZoom+canvas.height/2+viewPanY, sz:vz };
}
function unproject(sx:number,sy:number,sz=0): { x:number; y:number; z:number } {
  const vx=(sx-canvas.width/2-viewPanX)/viewZoom;
  const vy=-(sy-canvas.height/2-viewPanY)/viewZoom;
  const inv=mat3Transpose(viewRot);
  const [wx,wy,wz]=rotVec(inv,vx,vy,sz);
  return {x:wx,y:wy,z:wz};
}
function applyRotationDelta(dx:number,dy:number): void {
  const rx=dy*0.007, ry=dx*0.007;
  const cosX=Math.cos(rx),sinX=Math.sin(rx),cosY=Math.cos(ry),sinY=Math.sin(ry);
  const Rx=[1,0,0, 0,cosX,-sinX, 0,sinX,cosX];
  const Ry=[cosY,0,sinY, 0,1,0, -sinY,0,cosY];
  viewRot=mat3Mul(mat3Mul(Ry,Rx),viewRot);
}
function atomScreenR(el:string): number {
  return Math.max(5,Math.min(14,(ELEMENT_DATA[el]?.r??0.8)*viewZoom*0.52));
}

// ─── 3D Renderer ─────────────────────────────────────────────────────────────

function render3D(): void {
  const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H);
  if(!atoms.length){
    ctx.fillStyle='#3d4245'; ctx.font='11px IBM Plex Mono'; ctx.textAlign='center';
    ctx.fillText('Use the Atom tool or a ring template to start',W/2,H/2);
    ctx.font='10px IBM Plex Mono';
    ctx.fillText('Drag to rotate · Scroll to zoom',W/2,H/2+18);
    return;
  }
  const proj=atoms.map(a=>({...project(a),atom:a}));
  const order=proj.map((_,i)=>i).sort((a,b)=>proj[a].sz-proj[b].sz);
  ctx.lineCap='round';
  for(const b of bonds){
    const pi=proj.find(p=>p.atom.id===b.a1), pj=proj.find(p=>p.atom.id===b.a2);
    if(!pi||!pj) continue;
    const stroke=b.aromatic?'#c8a060':'#8a9090';
    if(b.order===2){const{ox,oy}=perpUnit(pi.sx,pi.sy,pj.sx,pj.sy,3.5);drawBondLine(pi.sx-ox,pi.sy-oy,pj.sx-ox,pj.sy-oy,stroke,2,'');drawBondLine(pi.sx+ox,pi.sy+oy,pj.sx+ox,pj.sy+oy,stroke,2,'');}
    else if(b.order===3){const{ox,oy}=perpUnit(pi.sx,pi.sy,pj.sx,pj.sy,5);drawBondLine(pi.sx,pi.sy,pj.sx,pj.sy,stroke,2,'');drawBondLine(pi.sx-ox,pi.sy-oy,pj.sx-ox,pj.sy-oy,stroke,1.5,'');drawBondLine(pi.sx+ox,pi.sy+oy,pj.sx+ox,pj.sy+oy,stroke,1.5,'');}
    else drawBondLine(pi.sx,pi.sy,pj.sx,pj.sy,stroke,2,b.aromatic?'5,3':'');
  }
  if(dragMode==='bond'&&bondDragStart!==null&&bondPreviewPx){
    const sp=proj.find(p=>p.atom.id===bondDragStart);
    if(sp){ctx.setLineDash([4,3]);ctx.strokeStyle='#00c4a7';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(sp.sx,sp.sy);ctx.lineTo(bondPreviewPx.x,bondPreviewPx.y);ctx.stroke();ctx.setLineDash([]);}
  }
  for(const i of order){
    const{sx,sy,sz,atom:a}=proj[i];
    const col=ELEMENT_DATA[a.el]?.color??'#aaaaaa';
    const r=atomScreenR(a.el);
    const dim=1.0-Math.max(0,Math.min(0.4,-sz*0.035));
    const isSel=a.id===selAtomId, isHover=dragMode==='bond'&&bondHoverAtom===a.id;
    if(isSel||isHover){
      ctx.beginPath();ctx.arc(sx,sy,r+5,0,2*Math.PI);
      ctx.fillStyle=isSel?'rgba(0,196,167,.22)':'rgba(91,141,238,.22)';ctx.fill();
      ctx.strokeStyle=isSel?'#00c4a7':'#5b8dee';ctx.lineWidth=1.5;ctx.stroke();
    }
    const grad=ctx.createRadialGradient(sx-r*0.3,sy-r*0.3,r*0.05,sx,sy,r);
    grad.addColorStop(0,shadeColor(col,Math.min(1,dim*1.55)));
    grad.addColorStop(1,shadeColor(col,dim*0.55));
    ctx.beginPath();ctx.arc(sx,sy,r,0,2*Math.PI);ctx.fillStyle=grad;ctx.fill();
    if(a.el!=='C'||isSel||a.charge!==0){
      ctx.fillStyle='#fff';ctx.font=`bold ${Math.round(r*0.98)}px IBM Plex Mono`;
      ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(a.el,sx,sy+0.5);ctx.textBaseline='alphabetic';
    }
    if(a.charge!==0){
      ctx.fillStyle=a.charge>0?'#7aadff':'#ff7a7a';ctx.font='9px IBM Plex Mono';ctx.textAlign='left';
      ctx.fillText(a.charge>0?(a.charge>1?`+${a.charge}`:'+'): (a.charge<-1?`${a.charge}`:'−'),sx+r*0.6,sy-r*0.5);
    }
  }
}

function drawBondLine(x1:number,y1:number,x2:number,y2:number,stroke:string,lw:number,dash:string): void {
  ctx.setLineDash(dash?dash.split(',').map(Number):[]);
  ctx.strokeStyle=stroke;ctx.lineWidth=lw;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.setLineDash([]);
}
function perpUnit(x1:number,y1:number,x2:number,y2:number,d:number): {ox:number;oy:number} {
  const dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy)||1;return{ox:-dy/len*d,oy:dx/len*d};
}
function shadeColor(hex:string,f:number): string {
  const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
  const c=(v:number)=>Math.max(0,Math.min(255,Math.round(v)));
  return `rgb(${c(r*f)},${c(g*f)},${c(b*f)})`;
}

// ─── Atom / bond picking ──────────────────────────────────────────────────────

function pickAtom(sx:number,sy:number): number|null {
  let best:number|null=null,bestZ=-Infinity;
  for(const a of atoms){
    const p=project(a);
    if(Math.hypot(sx-p.sx,sy-p.sy)<=atomScreenR(a.el)+4&&p.sz>bestZ){bestZ=p.sz;best=a.id;}
  }
  return best;
}
function pickBond(sx:number,sy:number): number|null {
  let best:number|null=null,bestD=9;
  for(const b of bonds){
    const ai=atoms.find(a=>a.id===b.a1),aj=atoms.find(a=>a.id===b.a2);if(!ai||!aj) continue;
    const pi=project(ai),pj=project(aj);
    const d=ptSegDist(sx,sy,pi.sx,pi.sy,pj.sx,pj.sy);
    if(d<bestD){bestD=d;best=b.id;}
  }
  return best;
}
function ptSegDist(px:number,py:number,ax:number,ay:number,bx:number,by:number): number {
  const dx=bx-ax,dy=by-ay,l2=dx*dx+dy*dy;
  if(!l2) return Math.hypot(px-ax,py-ay);
  const t=Math.max(0,Math.min(1,((px-ax)*dx+(py-ay)*dy)/l2));
  return Math.hypot(px-(ax+t*dx),py-(ay+t*dy));
}

// ─── 2D Layout ────────────────────────────────────────────────────────────────

function normalizeAngle(a:number): number { while(a>Math.PI)a-=2*Math.PI;while(a<-Math.PI)a+=2*Math.PI;return a; }

function findCycles(ids:number[],adj:Map<number,number[]>): number[][] {
  const cycles:number[][]=[], seen=new Set<string>(), depth=new Map<number,number>(), par=new Map<number,number>();
  function dfs(v:number,d:number): void {
    depth.set(v,d);
    for(const u of adj.get(v)??[]){
      if(par.get(v)===u) continue;
      if(depth.has(u)&&depth.get(u)!<d){
        const cyc:number[]=[v];let x=v,g=0;while(x!==u&&g++<20){x=par.get(x)??u;cyc.push(x);}
        if(cyc.length>=3&&cyc.length<=8){const key=[...cyc].sort((a,b)=>a-b).join(',');if(!seen.has(key)){seen.add(key);cycles.push(cyc);}}
      } else if(!depth.has(u)){par.set(u,v);dfs(u,d+1);}
    }
  }
  for(const id of ids) if(!depth.has(id)){par.set(id,-1);dfs(id,0);}
  return cycles;
}
function placeFirstRing(ring:number[],pos:Map<number,Pos2D>,cx:number,cy:number): void {
  const N=ring.length,R=BL2/(2*Math.sin(Math.PI/N)),start=N%2===0?0:-Math.PI/2;
  for(let i=0;i<N;i++){const ang=start+i*(2*Math.PI/N);pos.set(ring[i],{x:cx+R*Math.cos(ang),y:cy+R*Math.sin(ang)});}
}
function placeRingOnEdge(ring:number[],pos:Map<number,Pos2D>): boolean {
  const N=ring.length;let aIdx=-1;
  for(let i=0;i<N;i++) if(pos.has(ring[i])&&pos.has(ring[(i+1)%N])){aIdx=i;break;}
  if(aIdx<0) return false;
  const pA=pos.get(ring[aIdx])!,pB=pos.get(ring[(aIdx+1)%N])!;
  const mx=(pA.x+pB.x)/2,my=(pA.y+pB.y)/2,dx=pB.x-pA.x,dy=pB.y-pA.y,len=Math.hypot(dx,dy)||BL2;
  const px=-dy/len,py=dx/len,H=BL2/(2*Math.tan(Math.PI/N));
  const cands=[{x:mx+px*H,y:my+py*H},{x:mx-px*H,y:my-py*H}];
  const all=[...pos.values()];
  const cxA=all.reduce((s,p)=>s+p.x,0)/all.length,cyA=all.reduce((s,p)=>s+p.y,0)/all.length;
  const d0=(cands[0].x-cxA)**2+(cands[0].y-cyA)**2,d1=(cands[1].x-cxA)**2+(cands[1].y-cyA)**2;
  const{x:rcx,y:rcy}=d0>d1?cands[0]:cands[1];
  const R=BL2/(2*Math.sin(Math.PI/N));
  const angA=Math.atan2(pA.y-rcy,pA.x-rcx),angB=Math.atan2(pB.y-rcy,pB.x-rcx);
  const nomStep=2*Math.PI/N,diff=normalizeAngle(angB-angA);
  const dir=Math.abs(normalizeAngle(diff-nomStep))<Math.abs(normalizeAngle(diff+nomStep))?1:-1;
  for(let k=0;k<N;k++){const id=ring[(aIdx+k)%N];if(!pos.has(id))pos.set(id,{x:rcx+R*Math.cos(angA+k*dir*nomStep),y:rcy+R*Math.sin(angA+k*dir*nomStep)});}
  return true;
}
function compute2DCoords(): Map<number,Pos2D> {
  const pos=new Map<number,Pos2D>();if(!atoms.length) return pos;
  const ids=atoms.map(a=>a.id),adj=new Map<number,number[]>();
  for(const a of atoms) adj.set(a.id,[]);
  for(const b of bonds){adj.get(b.a1)?.push(b.a2);adj.get(b.a2)?.push(b.a1);}
  const rings=findCycles(ids,adj);
  if(rings.length){rings.sort((a,b)=>a.length-b.length);placeFirstRing(rings[0],pos,0,0);let ch=true;while(ch){ch=false;for(const r of rings){const n=r.filter(id=>pos.has(id)).length;if(n>=2&&n<r.length)if(placeRingOnEdge(r,pos))ch=true;}}}
  if(!pos.size) pos.set(ids[0],{x:0,y:0});
  let safety=0;
  while(pos.size<ids.length&&safety++<2000){
    let ext=false;
    for(const a of atoms){
      if(pos.has(a.id)) continue;
      const placed=(adj.get(a.id)??[]).filter(n=>pos.has(n));if(!placed.length) continue;
      const nb=placed[0],pNb=pos.get(nb)!;
      const ea=(adj.get(nb)??[]).filter(n=>pos.has(n)&&n!==a.id).map(n=>Math.atan2(pos.get(n)!.y-pNb.y,pos.get(n)!.x-pNb.x));
      let best:number;
      if(!ea.length){best=0;}
      else if(ea.length===1){const c1=ea[0]+2*Math.PI/3,c2=ea[0]-2*Math.PI/3;const sc=(ang:number)=>{const px=pNb.x+BL2*Math.cos(ang),py=pNb.y+BL2*Math.sin(ang);return[...pos.values()].reduce((m,p)=>Math.min(m,(p.x-px)**2+(p.y-py)**2),Infinity);};best=sc(c1)>=sc(c2)?c1:c2;}
      else{const sorted=[...ea].sort((a,b)=>a-b);let maxG=-Infinity,mid=0;for(let i=0;i<sorted.length;i++){const next=sorted[(i+1)%sorted.length];let g=next-sorted[i];if(i===sorted.length-1)g+=2*Math.PI;if(g>maxG){maxG=g;mid=sorted[i]+g/2;}}best=mid;}
      pos.set(a.id,{x:pNb.x+BL2*Math.cos(best),y:pNb.y+BL2*Math.sin(best)});ext=true;
    }
    if(!ext) break;
  }
  const xs=[...pos.values()].map(p=>p.x),ys=[...pos.values()].map(p=>p.y);
  const ox=(Math.min(...xs)+Math.max(...xs))/2,oy=(Math.min(...ys)+Math.max(...ys))/2;
  for(const[id,p]of pos) pos.set(id,{x:p.x-ox,y:p.y-oy});
  return pos;
}

// ─── 2D Renderer ─────────────────────────────────────────────────────────────

const SVG_NS       = 'http://www.w3.org/2000/svg';
const DEP_BOND_COL = '#c4c8cc';
const DEP_BG       = '#0d0e0f';
const DBL_OFFSET   = BL2*0.12;
const WEDGE_Z_THRESH = 0.25;

function render2D(): void {
  depBonds.innerHTML='';depAtoms.innerHTML='';
  const pos=compute2DCoords();if(!pos.size) return;
  const W=depSvg.clientWidth||316,H=depSvg.clientHeight||220;
  const xs=[...pos.values()].map(p=>p.x),ys=[...pos.values()].map(p=>p.y);
  const spread=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys),BL2);
  const margin=28,scale=Math.min((W-margin*2)/spread,(H-margin*2)/spread,2.2);
  function s(p:Pos2D){return{x:p.x*scale+W/2,y:p.y*scale+H/2};}
  const adjM=new Map<number,number[]>();
  for(const a of atoms) adjM.set(a.id,[]);
  for(const b of bonds){adjM.get(b.a1)?.push(b.a2);adjM.get(b.a2)?.push(b.a1);}
  const rings2d=findCycles(atoms.map(a=>a.id),adjM);
  function dblOff(b:LbBond):{ox:number;oy:number}{
    const p1=s(pos.get(b.a1)!),p2=s(pos.get(b.a2)!);
    const dx=p2.x-p1.x,dy=p2.y-p1.y,len=Math.hypot(dx,dy)||1,nx=-dy/len,ny=dx/len;
    const ring=rings2d.find(r=>r.includes(b.a1)&&r.includes(b.a2));
    if(ring){const cxR=ring.reduce((s,id)=>s+(pos.get(id)?.x??0),0)/ring.length,cyR=ring.reduce((s,id)=>s+(pos.get(id)?.y??0),0)/ring.length;const mx=(pos.get(b.a1)!.x+pos.get(b.a2)!.x)/2,my=(pos.get(b.a1)!.y+pos.get(b.a2)!.y)/2;const sign=(cxR-mx)*nx+(cyR-my)*ny>0?1:-1;return{ox:nx*sign*DBL_OFFSET,oy:ny*sign*DBL_OFFSET};}
    return{ox:nx*DBL_OFFSET,oy:ny*DBL_OFFSET};
  }
  function stereo(b:LbBond):'none'|'wedge'|'dash'{
    if(b.order!==1||b.aromatic) return 'none';
    const ai=atoms.find(a=>a.id===b.a1),aj=atoms.find(a=>a.id===b.a2);if(!ai||!aj) return 'none';
    const dz=aj.z-ai.z;return Math.abs(dz)<WEDGE_Z_THRESH?'none':(dz>0?'wedge':'dash');
  }
  for(const b of bonds){
    const p1r=pos.get(b.a1),p2r=pos.get(b.a2);if(!p1r||!p2r) continue;
    const p1=s(p1r),p2=s(p2r),st=stereo(b);
    if(st==='wedge'){drawWedge(p1,p2,DEP_BOND_COL);}
    else if(st==='dash'){drawDash(p1,p2,DEP_BOND_COL);}
    else if(b.order===2){const{ox,oy}=dblOff(b),trim=0.18;depBonds.appendChild(makeLine2D(p1.x,p1.y,p2.x,p2.y,DEP_BOND_COL,1.5,''));depBonds.appendChild(makeLine2D(p1.x+(p2.x-p1.x)*trim+ox,p1.y+(p2.y-p1.y)*trim+oy,p2.x-(p2.x-p1.x)*trim+ox,p2.y-(p2.y-p1.y)*trim+oy,DEP_BOND_COL,1.5,''));}
    else if(b.order===3){const{ox,oy}=dblOff(b);depBonds.appendChild(makeLine2D(p1.x,p1.y,p2.x,p2.y,DEP_BOND_COL,1.5,''));depBonds.appendChild(makeLine2D(p1.x-ox,p1.y-oy,p2.x-ox,p2.y-oy,DEP_BOND_COL,1.0,''));depBonds.appendChild(makeLine2D(p1.x+ox,p1.y+oy,p2.x+ox,p2.y+oy,DEP_BOND_COL,1.0,''));}
    else depBonds.appendChild(makeLine2D(p1.x,p1.y,p2.x,p2.y,DEP_BOND_COL,1.5,b.aromatic?'4,3':''));
  }
  for(const a of atoms){
    const pr=pos.get(a.id);if(!pr) continue;
    const p=s(pr),hc=implicitH(a);
    if(a.el==='C'&&(adjM.get(a.id)?.length??0)>0&&a.charge===0) continue;
    const col=ELEMENT_DATA[a.el]?.color??'#e2e4e6';
    const nbs=adjM.get(a.id)??[];
    const xSum=nbs.reduce((s,n)=>s+((pos.get(n)?.x??0)-pr.x),0);
    const hLeft=xSum>0;
    let label=a.el;
    if(hc>0){const h='H'+(hc>1?hc:'');label=hLeft?h+a.el:a.el+h;}
    if(a.charge!==0) label+=a.charge>0?(a.charge>1?`+${a.charge}`:'+'): (a.charge<-1?`${a.charge}`:'−');
    const fs=11,charW=fs*0.62,lw=label.length*charW,lh=fs+2;
    const bg=document.createElementNS(SVG_NS,'rect');
    bg.setAttribute('x',String(p.x-lw/2-2));bg.setAttribute('y',String(p.y-lh/2-1));
    bg.setAttribute('width',String(lw+4));bg.setAttribute('height',String(lh+2));
    bg.setAttribute('fill',DEP_BG);depAtoms.appendChild(bg);
    const txt=document.createElementNS(SVG_NS,'text');
    txt.setAttribute('x',String(p.x));txt.setAttribute('y',String(p.y+fs*0.36));
    txt.setAttribute('text-anchor','middle');txt.setAttribute('font-family','IBM Plex Mono,monospace');
    txt.setAttribute('font-size',String(fs));txt.setAttribute('font-weight','500');
    txt.setAttribute('fill',col);txt.setAttribute('pointer-events','none');
    txt.textContent=label;depAtoms.appendChild(txt);
  }
}

function makeLine2D(x1:number,y1:number,x2:number,y2:number,stroke:string,sw:number,dash:string): SVGLineElement {
  const l=document.createElementNS(SVG_NS,'line');
  l.setAttribute('x1',String(x1));l.setAttribute('y1',String(y1));
  l.setAttribute('x2',String(x2));l.setAttribute('y2',String(y2));
  l.setAttribute('stroke',stroke);l.setAttribute('stroke-width',String(sw));
  l.setAttribute('stroke-linecap','round');if(dash)l.setAttribute('stroke-dasharray',dash);
  return l;
}
function drawWedge(p1:Pos2D,p2:Pos2D,color:string): void {
  const dx=p2.x-p1.x,dy=p2.y-p1.y,len=Math.hypot(dx,dy)||1,nx=-dy/len,ny=dx/len,w=5;
  const poly=document.createElementNS(SVG_NS,'polygon');
  poly.setAttribute('points',`${p1.x},${p1.y} ${p2.x+nx*w},${p2.y+ny*w} ${p2.x-nx*w},${p2.y-ny*w}`);
  poly.setAttribute('fill',color);depBonds.appendChild(poly);
}
function drawDash(p1:Pos2D,p2:Pos2D,color:string): void {
  const STEPS=6,dx=p2.x-p1.x,dy=p2.y-p1.y,len=Math.hypot(dx,dy)||1,nx=-dy/len,ny=dx/len;
  for(let i=1;i<=STEPS;i++){const t=i/STEPS,cx=p1.x+t*dx,cy=p1.y+t*dy,hw=t*5;depBonds.appendChild(makeLine2D(cx-nx*hw,cy-ny*hw,cx+nx*hw,cy+ny*hw,color,1.5,''));}
}

// ─── Chemistry helpers ────────────────────────────────────────────────────────

function atomById(id:number): LbAtom|undefined { return atoms.find(a=>a.id===id); }
function bondsOf(id:number):  LbBond[]         { return bonds.filter(b=>b.a1===id||b.a2===id); }
function implicitH(atom:LbAtom): number {
  if(atom.el==='H') return 0;
  const base=({C:4,N:3,O:2,S:2,P:3,F:1,Cl:1,Br:1,I:1,B:3,Si:4} as Record<string,number>)[atom.el]??0;
  const used=bondsOf(atom.id).reduce((s,b)=>s+(b.aromatic?1.5:b.order),0);
  return Math.max(0,Math.round(base-used+atom.charge));
}
function formula(): string {
  const cnt:Record<string,number>={};
  for(const a of atoms) cnt[a.el]=(cnt[a.el]??0)+1;
  for(const a of atoms){const h=implicitH(a);if(h>0)cnt['H']=(cnt['H']??0)+h;}
  let s='';if(cnt['C']){s+='C';if(cnt['C']>1)s+=cnt['C'];delete cnt['C'];}if(cnt['H']){s+='H';if(cnt['H']>1)s+=cnt['H'];delete cnt['H'];}
  for(const el of Object.keys(cnt).sort()){s+=el;if(cnt[el]>1)s+=cnt[el];}return s||'—';
}
function molecularWeight(): string {
  let m=0;for(const a of atoms)m+=(MW_TABLE[a.el]??0);for(const a of atoms)m+=(MW_TABLE['H']??1.008)*implicitH(a);
  return m>0?m.toFixed(2):'—';
}
function toSmiles(): string {
  if(!atoms.length) return '';
  const vis=new Set<number>(),bvis=new Set<number>(),rn:Record<number,number>={};let nr=1;
  function symB(b:LbBond){return b.aromatic?'':(b.order===2?'=':(b.order===3?'#':''));}
  function dfs(id:number,par:number|null):string{
    vis.add(id);const a=atomById(id)!;
    let s=a.charge!==0?`[${a.el}${a.charge>0?(a.charge>1?`+${a.charge}`:'+'): (a.charge<-1?`${a.charge}`:'-')}]`:a.el;
    const br:string[]=[];
    for(const b of bondsOf(id)){const o=b.a1===id?b.a2:b.a1;if(o===par)continue;if(bvis.has(b.id))continue;bvis.add(b.id);if(vis.has(o)){if(rn[b.id]===undefined)rn[b.id]=nr++;s+=symB(b)+rn[b.id];}else br.push(symB(b)+dfs(o,id));}
    if(!br.length) return s;const main=br.pop()!;return s+br.map(b=>`(${b})`).join('')+main;
  }
  return dfs(atoms[0].id,null);
}
function serializeMol() {
  const charge=parseInt((document.getElementById('mol-charge') as HTMLInputElement).value,10)||0;
  const mult  =parseInt((document.getElementById('mol-mult')   as HTMLInputElement).value,10)||1;
  return {atoms:atoms.map(a=>({id:a.id,element:a.el,x:a.x,y:a.y,charge:a.charge,radical:a.radical})),bonds:bonds.map(b=>({atom1:b.a1,atom2:b.a2,order:b.order,aromatic:b.aromatic})),charge,multiplicity:mult};
}
function serializeCoords() { return atoms.map(a=>({x:a.x,y:a.y,z:a.z})); }

// ─── Undo / redo ──────────────────────────────────────────────────────────────

function snapshot(): string { return JSON.stringify({atoms,bonds,nextAtomId,nextBondId}); }
function saveSnapshot(): void { undoStack.push(snapshot());if(undoStack.length>MAX_UNDO)undoStack.shift();redoStack.length=0; }
function applySnapshot(s:string): void { ({atoms,bonds,nextAtomId,nextBondId}=JSON.parse(s) as any);selAtomId=null; }
function undo(): void { if(!undoStack.length)return;redoStack.push(snapshot());applySnapshot(undoStack.pop()!);render(); }
function redo(): void { if(!redoStack.length)return;undoStack.push(snapshot());applySnapshot(redoStack.pop()!);render(); }

// ─── Render ───────────────────────────────────────────────────────────────────

function render(): void { render3D();render2D();updateStats(); }
function updateStats(): void {
  document.getElementById('stat-atoms')!  .textContent=String(atoms.filter(a=>a.el!=='H').length);
  document.getElementById('stat-bonds')!  .textContent=String(bonds.length);
  document.getElementById('stat-formula')!.textContent=formula();
  document.getElementById('stat-mw')!     .textContent=molecularWeight()+' g/mol';
  document.getElementById('smiles-out')!  .textContent=toSmiles()||'(empty)';
}

// ─── Minimization UI helpers ──────────────────────────────────────────────────

function setMinimizeBusy(busy: boolean): void {
  minimizeRunning = busy;
  const btnRun    = document.getElementById('btn-minimize-full')   as HTMLButtonElement;
  const btnQuick  = document.getElementById('btn-minimize-quick')  as HTMLButtonElement;
  const btnCancel = document.getElementById('btn-minimize-cancel') as HTMLButtonElement;
  const progWrap  = document.getElementById('min-progress-wrap')!;
  btnRun.disabled   = busy;
  btnQuick.disabled = busy;
  btnCancel.style.display = busy ? 'flex' : 'none';
  progWrap.style.display  = busy ? 'block' : 'none';
  if (!busy) {
    document.getElementById('min-progress-fill')!.style.width = '0%';
    (document.getElementById('min-progress-fill')! as HTMLElement).classList.remove('indeterminate');
  }
}

function onMinProgress(p: MinProgressPayload): void {
  const fill = document.getElementById('min-progress-fill')! as HTMLElement;
  const text = document.getElementById('min-progress-text')!;
  if (p.total > 0) {
    // Determinate — UFF
    fill.classList.remove('indeterminate');
    fill.style.width = `${Math.min(100, (p.step / p.total) * 100).toFixed(1)}%`;
  } else {
    // Indeterminate — external process
    if (!fill.classList.contains('indeterminate')) fill.classList.add('indeterminate');
  }
  text.textContent = p.message;
}

function onMinDone(result: MinDonePayload): void {
  result.coords.forEach((c,i) => { if(atoms[i]){atoms[i].x=c.x;atoms[i].y=c.y;atoms[i].z=c.z;} });
  render();
  setMinimizeBusy(false);
  setStatus('min-status', `✓ ${result.method} · ${result.steps_run} cycles · E = ${result.final_energy.toFixed(4)} kcal/mol`, 'status-ok');
}

// ─── Atom / bond operations ───────────────────────────────────────────────────

function deleteAtom(aid:number): void {
  atoms=atoms.filter(a=>a.id!==aid);bonds=bonds.filter(b=>b.a1!==aid&&b.a2!==aid);
  if(selAtomId===aid)selAtomId=null;render();
}
function cycleBondOrder(bid:number): void {
  const b=bonds.find(x=>x.id===bid);if(!b)return;
  if(b.aromatic){b.aromatic=false;b.order=1;}else if(b.order===1)b.order=2;else if(b.order===2)b.order=3;else b.order=1;
  render();
}
function completeBond(a1:number,a2:number): void {
  const ex=bonds.find(b=>(b.a1===a1&&b.a2===a2)||(b.a2===a1&&b.a1===a2));
  if(ex){ex.order=bondOrder==='a'?1:bondOrder;ex.aromatic=bondOrder==='a';}
  else bonds.push({id:nextBondId++,a1,a2,order:bondOrder==='a'?1:bondOrder,aromatic:bondOrder==='a'});
}

// ─── Ring / SMILES ────────────────────────────────────────────────────────────

function insertRing(ringId:string): void {
  const r=RINGS[ringId];if(!r) return;
  saveSnapshot();
  const off=atoms.length?{x:Math.max(...atoms.map(a=>a.x))+BL3*r.n*0.6,y:0,z:0}:{x:0,y:0,z:0};
  const R=BL3/(2*Math.sin(Math.PI/r.n)),start=r.n%2===0?0:-Math.PI/2;
  const newA:LbAtom[]=[];
  for(let k=0;k<r.n;k++){const ang=start+k*(2*Math.PI/r.n);newA.push({id:nextAtomId++,el:r.elems[k],x:off.x+R*Math.cos(ang),y:off.y+R*Math.sin(ang),z:0,charge:0,radical:0});}
  for(let k=0;k<r.n;k++) bonds.push({id:nextBondId++,a1:newA[k].id,a2:newA[(k+1)%r.n].id,order:r.bonds[k]??1,aromatic:!!(r.aromatic&&r.bonds[k]===1)});
  atoms.push(...newA);render();
}
function parseSMILES(s:string):{atoms:LbAtom[];bonds:LbBond[]} {
  const na:LbAtom[]=[],nb:LbBond[]=[]; let aid=1,bid=1,i=0,prev=-1,pO=1,pA=false;
  const stk:number[]=[],ro:Record<string,{idx:number;order:number;aro:boolean}>={};
  while(i<s.length){
    const ch=s[i];
    if(ch==='('){stk.push(prev);i++;continue;}if(ch===')'){prev=stk.pop()??-1;i++;continue;}
    if(ch==='-'){pO=1;i++;continue;}if(ch==='='){pO=2;i++;continue;}if(ch==='#'){pO=3;i++;continue;}if(ch===':'){pA=true;i++;continue;}
    if(/\d/.test(ch)){const d=ch;i++;if(ro[d]){const{idx,order,aro}=ro[d];nb.push({id:bid++,a1:na[idx].id,a2:na[prev].id,order:order||pO,aromatic:aro||pA});delete ro[d];}else ro[d]={idx:prev,order:pO,aro:pA};pO=1;pA=false;continue;}
    let el='',charge=0;
    if(ch==='['){i++;let ins='';while(i<s.length&&s[i]!==']')ins+=s[i++];i++;const m=ins.match(/^([A-Z][a-z]?)(?:H\d?)?([+-]\d*)?$/i);if(m){el=m[1];const cp=m[2]??'';charge=cp===''?0:cp==='+'?1:cp==='-'?-1:parseInt(cp,10);}else el=ins.replace(/[^A-Za-z]/g,'').slice(0,2)||'C';}
    else if(/[A-Z]/.test(ch)){el=ch;i++;if(i<s.length&&/[a-z]/.test(s[i])&&['l','r','i','e'].includes(s[i])){el+=s[i];i++;}}
    else if(/[a-z]/.test(ch)){el=ch.toUpperCase();pA=true;i++;}else{i++;continue;}
    if(!el) continue;
    const ci=na.length;na.push({id:aid++,el,x:0,y:0,z:0,charge,radical:0});
    if(prev>=0) nb.push({id:bid++,a1:na[prev].id,a2:na[ci].id,order:pO,aromatic:pA});
    prev=ci;pO=1;pA=false;
  }
  layout3D(na,nb);return{atoms:na,bonds:nb};
}
function layout3D(na:LbAtom[],nb:LbBond[]): void {
  if(!na.length) return;
  const adj=new Map<number,number[]>();for(const a of na) adj.set(a.id,[]);for(const b of nb){adj.get(b.a1)?.push(b.a2);adj.get(b.a2)?.push(b.a1);}
  const pos=new Map<number,{x:number,y:number}>(),vis=new Set([na[0].id]);pos.set(na[0].id,{x:0,y:0});
  const q=[{id:na[0].id,dir:0}];
  while(q.length){const{id,dir}=q.shift()!;const nbs=(adj.get(id)??[]).filter(n=>!vis.has(n));let ang=dir+Math.PI/6;for(const n of nbs){pos.set(n,{x:(pos.get(id)?.x??0)+Math.cos(ang)*BL3,y:(pos.get(id)?.y??0)+Math.sin(ang)*BL3});vis.add(n);q.push({id:n,dir:ang});ang-=Math.PI/3;}}
  for(const a of na){const p=pos.get(a.id)??{x:0,y:0};a.x=p.x;a.y=p.y;a.z=0;}
}

// ─── Atom props panel ─────────────────────────────────────────────────────────

function showAtomProps(aid:number): void {
  selAtomId=aid;const a=atomById(aid),panel=document.getElementById('atom-props-panel')!;
  if(!a){panel.innerHTML='<span style="color:var(--muted)">No atom selected</span>';return;}
  panel.innerHTML=`
    <div class="form-row"><span class="form-lbl">Element</span><input class="form-input narrow" id="ap-el" value="${a.el}"/></div>
    <div class="form-row"><span class="form-lbl">Charge</span><input class="form-input narrow" id="ap-charge" type="number" value="${a.charge}"/></div>
    <div class="form-row" style="gap:4px">
      <input class="form-input" id="ap-x" type="number" step="0.01" value="${a.x.toFixed(3)}" style="width:58px;flex:none" title="x (Å)"/>
      <input class="form-input" id="ap-y" type="number" step="0.01" value="${a.y.toFixed(3)}" style="width:58px;flex:none" title="y (Å)"/>
      <input class="form-input" id="ap-z" type="number" step="0.01" value="${a.z.toFixed(3)}" style="width:58px;flex:none" title="z (Å)"/>
    </div>
    <div style="font-size:10px;color:var(--muted);margin:4px 0">Implicit H: ${implicitH(a)} · Bonds: ${bondsOf(aid).length}</div>
    <button class="action-btn secondary" id="ap-del" style="margin-top:6px;font-size:10px">Delete atom</button>`;
  const upd=()=>{saveSnapshot();render();};
  (document.getElementById('ap-el')    as HTMLInputElement).addEventListener('change',()=>{a.el=(document.getElementById('ap-el') as HTMLInputElement).value.trim();upd();});
  (document.getElementById('ap-charge')as HTMLInputElement).addEventListener('change',()=>{a.charge=parseInt((document.getElementById('ap-charge') as HTMLInputElement).value,10)||0;upd();});
  (document.getElementById('ap-x')     as HTMLInputElement).addEventListener('change',()=>{a.x=parseFloat((document.getElementById('ap-x') as HTMLInputElement).value)||0;upd();});
  (document.getElementById('ap-y')     as HTMLInputElement).addEventListener('change',()=>{a.y=parseFloat((document.getElementById('ap-y') as HTMLInputElement).value)||0;upd();});
  (document.getElementById('ap-z')     as HTMLInputElement).addEventListener('change',()=>{a.z=parseFloat((document.getElementById('ap-z') as HTMLInputElement).value)||0;upd();});
  document.getElementById('ap-del')!.addEventListener('click',()=>{saveSnapshot();deleteAtom(aid);panel.innerHTML='<span style="color:var(--muted)">Deleted</span>';});
}

// ─── Tool selection ───────────────────────────────────────────────────────────

function setTool(t:Tool): void {
  tool=t;
  (['select','atom','bond','erase'] as const).forEach(id=>document.getElementById(`tool-${id}`)!.classList.toggle('active',t===id));
}

// ─── Status helpers ───────────────────────────────────────────────────────────

function setStatus(id:string,msg:string,cls:'status-ok'|'status-err'|'status-info'): void {
  const el=document.getElementById(id)!;el.textContent=msg;el.className=`status-line ${cls}`;
}
function setBusy(btn:HTMLButtonElement,lbl:string): void { btn.disabled=true;btn.innerHTML=`<span class="spinner"></span> ${lbl}`; }
function setIdle(btn:HTMLButtonElement,lbl:string): void { btn.disabled=false;btn.textContent=lbl; }

// ─── Rust wrappers ────────────────────────────────────────────────────────────
//
// runMinimize() fires lb_minimize_start which returns immediately.
// Results arrive asynchronously via the lb:min-progress / lb:min-done / lb:min-error
// events registered in Step 1.  The GUI stays fully interactive throughout.

async function runMinimize(quick = false): Promise<void> {
  if (!atoms.length) { alert('No atoms to minimize.'); return; }
  if (minimizeRunning) return;

  const method  = quick ? 'uff' : (document.getElementById('min-method') as HTMLSelectElement).value;
  const steps   = quick ? 200  : (parseInt((document.getElementById('min-steps') as HTMLInputElement).value, 10) || 500);
  const conv    = quick ? 0.005 : (parseFloat((document.getElementById('min-conv') as HTMLInputElement).value) || 0.001);
  const xtbPath  = (document.getElementById('xtb-path')  as HTMLInputElement)?.value  || 'xtb';
  const xtbLevel = (document.getElementById('xtb-level') as HTMLSelectElement)?.value || 'normal';
  const orcaPath = (document.getElementById('orca-path') as HTMLInputElement)?.value  || 'orca';
  const orcaCores= parseInt((document.getElementById('orca-cores') as HTMLInputElement)?.value || '4', 10);

  setMinimizeBusy(true);
  document.getElementById('min-progress-text')!.textContent = 'Starting…';
  setStatus('min-status', '', 'status-info');

  try {
    // lb_minimize_start returns Ok(()) immediately; results come back via events
    await invoke('lb_minimize_start', {
      mol: serializeMol(), coords: serializeCoords(),
      method, steps, conv, xtbPath, xtbLevel, orcaPath, orcaCores,
    });
  } catch (err) {
    // Only fires if the command itself failed to dispatch (e.g. serialization error)
    setMinimizeBusy(false);
    setStatus('min-status', `✗ ${err}`, 'status-err');
  }
}

async function cancelMinimize(): Promise<void> {
  try { await invoke('lb_minimize_cancel'); } catch (_) {}
}

async function generateQmInput(): Promise<void> {
  const program=(document.getElementById('qm-program') as HTMLSelectElement).value;
  const jobtype=(document.getElementById('qm-jobtype') as HTMLSelectElement).value;
  const mS=(document.getElementById('qm-method') as HTMLSelectElement).value;
  const method=mS==='custom'?(document.getElementById('qm-method-custom') as HTMLInputElement).value:mS;
  const bS=(document.getElementById('qm-basis') as HTMLSelectElement).value;
  const basis=bS==='custom'?(document.getElementById('qm-basis-custom') as HTMLInputElement).value:bS;
  const solvent=(document.getElementById('qm-solvent') as HTMLSelectElement).value;
  const extra=(document.getElementById('qm-extra') as HTMLTextAreaElement).value.trim();
  const btn=document.getElementById('btn-gen-qm') as HTMLButtonElement;
  setBusy(btn,'Generating…');document.getElementById('qm-status')!.textContent='';
  try{
    const input=await invoke<string>('lb_export_qm_input',{mol:serializeMol(),coords:serializeCoords(),program,jobtype,method,basis,solvent,extra});
    (document.getElementById('qm-preview') as HTMLTextAreaElement).value=input;
    setStatus('qm-status','✓ Input file generated.','status-ok');
  }catch(e){setStatus('qm-status',`✗ ${e}`,'status-err');}
  finally{setIdle(btn,'Generate Input File');}
}

async function saveFile(name:string,ext:string,content:string,sid:string): Promise<void> {
  const path=await saveDialog({defaultPath:name,filters:[{name:'File',extensions:[ext]}]});
  if(!path) return;
  await invoke('lb_write_text',{path,content});
  setStatus(sid,`✓ Saved to ${path}`,'status-ok');
}
async function saveQmInput(): Promise<void> {
  const p=(document.getElementById('qm-program') as HTMLSelectElement).value;
  const ext=({qchem:'in',orca:'inp',gaussian:'gjf',nwchem:'nw',psi4:'py',gamess:'inp'} as Record<string,string>)[p]??'txt';
  await saveFile(`molecule.${ext}`,ext,(document.getElementById('qm-preview') as HTMLTextAreaElement).value,'qm-status');
}

// ─── Docking helpers ──────────────────────────────────────────────────────────

function setDockBusy(busy: boolean): void {
  dockRunning = busy;
  const btnRun    = document.getElementById('btn-run-dock')    as HTMLButtonElement;
  const btnCancel = document.getElementById('btn-cancel-dock') as HTMLButtonElement;
  const progWrap  = document.getElementById('dock-progress-wrap')!;
  btnRun.disabled         = busy;
  btnCancel.style.display = busy ? 'flex' : 'none';
  progWrap.style.display  = busy ? 'block' : 'none';
  if (!busy) {
    const fill = document.getElementById('dock-progress-fill')! as HTMLElement;
    fill.classList.remove('indeterminate');
  }
}

// Fetch receptor centre of mass and populate the cx/cy/cz fields.
// Called when the receptor path changes and box mode is 'receptor'.
async function fetchReceptorCenter(): Promise<void> {
  const receptorPath = (document.getElementById('dock-receptor-path') as HTMLInputElement).value.trim();
  if (!receptorPath) return;
  try {
    const c = await invoke<{ x:number; y:number; z:number }>('lb_get_receptor_center', { receptorPath });
    (document.getElementById('dock-cx') as HTMLInputElement).value = c.x.toFixed(2);
    (document.getElementById('dock-cy') as HTMLInputElement).value = c.y.toFixed(2);
    (document.getElementById('dock-cz') as HTMLInputElement).value = c.z.toFixed(2);
    setStatus('dock-status', `✓ Box centred on receptor CoM (${c.x.toFixed(1)}, ${c.y.toFixed(1)}, ${c.z.toFixed(1)})`, 'status-ok');
  } catch (e) {
    setStatus('dock-status', `⚠ Could not read receptor centre: ${e}`, 'status-info');
  }
}

function updateLigandCenter(): void {
  if (!atoms.length) return;
  const cx = atoms.reduce((s,a)=>s+a.x,0)/atoms.length;
  const cy = atoms.reduce((s,a)=>s+a.y,0)/atoms.length;
  const cz = atoms.reduce((s,a)=>s+a.z,0)/atoms.length;
  (document.getElementById('dock-cx') as HTMLInputElement).value = cx.toFixed(2);
  (document.getElementById('dock-cy') as HTMLInputElement).value = cy.toFixed(2);
  (document.getElementById('dock-cz') as HTMLInputElement).value = cz.toFixed(2);
}

function activeBoxMode(): string {
  return document.querySelector<HTMLButtonElement>('#dock-box-mode .radio-btn.active')?.dataset['box'] ?? 'receptor';
}

// Open the visualizer in dock_preview mode — shows receptor + search box.
async function openDockPreview(): Promise<void> {
  const receptorPath = (document.getElementById('dock-receptor-path') as HTMLInputElement).value.trim();
  if (!receptorPath) { setStatus('dock-status', '✗ Select a receptor file first.', 'status-err'); return; }
  const cx = parseFloat((document.getElementById('dock-cx') as HTMLInputElement).value) || 0;
  const cy = parseFloat((document.getElementById('dock-cy') as HTMLInputElement).value) || 0;
  const cz = parseFloat((document.getElementById('dock-cz') as HTMLInputElement).value) || 0;
  const sx = parseFloat((document.getElementById('dock-sx') as HTMLInputElement).value) || 20;
  const sy = parseFloat((document.getElementById('dock-sy') as HTMLInputElement).value) || 20;
  const sz = parseFloat((document.getElementById('dock-sz') as HTMLInputElement).value) || 20;
  const { WebviewWindow } = await import('@tauri-apps/api/webviewWindow');
  const params = new URLSearchParams({
    mode: 'dock_preview',
    receptor: receptorPath,
    cx: String(cx), cy: String(cy), cz: String(cz),
    sx: String(sx), sy: String(sy), sz: String(sz),
  });
  new WebviewWindow('dock-preview', {
    url:       `visualizer.html?${params.toString()}`,
    title:     'Dock Preview — Search Box',
    width:     960,
    height:    720,
    resizable: true,
    focus:     true,
  });
}

// Open the visualizer in docked_complex mode — shows receptor + selected pose.
// Row clicks in the pose table also emit viz:dock-pose events to update an
// already-open complex window.
async function openDockedComplex(): Promise<void> {
  if (!dockPoses.length || !dockOutputPdbqt) {
    setStatus('dock-status', '✗ Run docking first.', 'status-err'); return;
  }
  const { WebviewWindow } = await import('@tauri-apps/api/webviewWindow');
  const params = new URLSearchParams({
    mode:        'docked_complex',
    receptor:    dockReceptorPath,
    pdbqt:       dockOutputPdbqt,
    pose:        String(dockPoses[dockSelectedPose]?.rank ?? 1),
    total_poses: String(dockPoses.length),
  });
  new WebviewWindow('docked-complex', {
    url:       `visualizer.html?${params.toString()}`,
    title:     'Docked Complex',
    width:     960,
    height:    720,
    resizable: true,
    focus:     true,
  });
}

function onDockDone(result: DockDonePayload): void {
  dockPoses       = result.poses;
  dockSelectedPose = 0;
  dockOutputPdbqt  = result.output_pdbqt;
  setDockBusy(false);
  setStatus('dock-status',
    `✓ ${result.engine} · ${result.poses.length} poses · best = ${result.best_affinity.toFixed(2)} kcal/mol`,
    'status-ok');
  renderPoseTable();
  document.getElementById('dock-results-section')!.style.display = 'block';
}

function renderPoseTable(): void {
  const tbody = document.getElementById('dock-pose-tbody')!;
  tbody.innerHTML = '';
  dockPoses.forEach((pose, idx) => {
    const tr = document.createElement('tr');
    if (idx === dockSelectedPose) tr.classList.add('selected');
    tr.innerHTML = `
      <td>${pose.rank}</td>
      <td class="affinity-cell${idx === 0 ? ' best' : ''}">${pose.affinity.toFixed(2)}</td>
      <td>${pose.rmsd_lb.toFixed(2)}</td>`;
    tr.addEventListener('click', async () => {
      dockSelectedPose = idx;
      tbody.querySelectorAll('tr').forEach(r => r.classList.remove('selected'));
      tr.classList.add('selected');
      // If the complex viewer is open, update its pose
      try {
        const { emit } = await import('@tauri-apps/api/event');
        await emit('viz:dock-pose', { pose: pose.rank });
      } catch (_) {}
    });
    tbody.appendChild(tr);
  });
}

async function runDock(): Promise<void> {
  if (!atoms.length) { alert('Build a molecule first.'); return; }
  if (dockRunning) return;

  const receptorPath  = (document.getElementById('dock-receptor-path') as HTMLInputElement).value.trim();
  if (!receptorPath) { setStatus('dock-status', '✗ Select a receptor file first.', 'status-err'); return; }
  dockReceptorPath = receptorPath;

  const engine        = (document.getElementById('dock-engine')         as HTMLSelectElement).value;
  const binary        = (document.getElementById('dock-binary')         as HTMLInputElement).value.trim() || (engine === 'vina' ? 'vina' : 'unidock');
  const exhaustiveness= parseInt((document.getElementById('dock-exhaustiveness') as HTMLInputElement).value, 10) || 8;
  const numPoses      = parseInt((document.getElementById('dock-num-poses')      as HTMLInputElement).value, 10) || 9;
  const energyRange   = parseFloat((document.getElementById('dock-energy-range') as HTMLInputElement).value) || 3;
  const cpu           = parseInt((document.getElementById('dock-cpu')            as HTMLInputElement).value, 10) || 4;

  // Box center: resolve from mode
  const boxMode = activeBoxMode();
  let centerX: number, centerY: number, centerZ: number;
  if (boxMode === 'receptor') {
    centerX = parseFloat((document.getElementById('dock-cx') as HTMLInputElement).value) || 0;
    centerY = parseFloat((document.getElementById('dock-cy') as HTMLInputElement).value) || 0;
    centerZ = parseFloat((document.getElementById('dock-cz') as HTMLInputElement).value) || 0;
  } else if (boxMode === 'ligand') {
    centerX = atoms.reduce((s,a)=>s+a.x,0)/atoms.length;
    centerY = atoms.reduce((s,a)=>s+a.y,0)/atoms.length;
    centerZ = atoms.reduce((s,a)=>s+a.z,0)/atoms.length;
  } else {
    centerX = parseFloat((document.getElementById('dock-cx') as HTMLInputElement).value) || 0;
    centerY = parseFloat((document.getElementById('dock-cy') as HTMLInputElement).value) || 0;
    centerZ = parseFloat((document.getElementById('dock-cz') as HTMLInputElement).value) || 0;
  }
  const sizeX = parseFloat((document.getElementById('dock-sx') as HTMLInputElement).value) || 20;
  const sizeY = parseFloat((document.getElementById('dock-sy') as HTMLInputElement).value) || 20;
  const sizeZ = parseFloat((document.getElementById('dock-sz') as HTMLInputElement).value) || 20;

  setDockBusy(true);
  document.getElementById('dock-progress-text')!.textContent = 'Starting…';
  document.getElementById('dock-results-section')!.style.display = 'none';
  setStatus('dock-status', '', 'status-info');

  try {
    await invoke('lb_dock_start', {
      mol: serializeMol(), coords: serializeCoords(),
      engine, binary, receptorPath,
      obabelPath: (document.getElementById('dock-obabel-path') as HTMLInputElement).value.trim() || 'obabel',
      centerX, centerY, centerZ,
      sizeX, sizeY, sizeZ,
      exhaustiveness, numPoses, energyRange, cpu,
    });
  } catch(err) {
    setDockBusy(false);
    setStatus('dock-status', `✗ ${err}`, 'status-err');
  }
}

async function cancelDock(): Promise<void> {
  try { await invoke('lb_dock_cancel'); } catch (_) {}
}

// ─── Main ─────────────────────────────────────────────────────────────────────
//
// Mirrors initVisualizer() exactly:
//   1. Assign DOM references, size canvas.
//   2. Register backend event listeners FIRST (before any async work).
//   3. Populate element grid.
//   4. Wire unified pointer events.
//   5. Wire toolbar buttons.
//   6. Wire right-panel controls.
//   7. Initial render.

async function initLigandBuilder(): Promise<void> {

  // ── Step 1: DOM references ─────────────────────────────────────────────────
  canvas   = document.getElementById('editor-canvas') as HTMLCanvasElement;
  ctx      = canvas.getContext('2d')!;
  depSvg   = document.getElementById('depiction-svg')  as unknown as SVGSVGElement;
  depBonds = document.getElementById('dep-bond-layer') as unknown as SVGGElement;
  depAtoms = document.getElementById('dep-atom-layer') as unknown as SVGGElement;
  tooltip  = document.getElementById('atom-tooltip')  as HTMLDivElement;

  const wrap = document.getElementById('editor-wrap')!;
  function resizeCanvas(): void { canvas.width=wrap.clientWidth;canvas.height=wrap.clientHeight;render(); }
  new ResizeObserver(resizeCanvas).observe(wrap);
  resizeCanvas();

  // ── Step 2: Register backend event listeners FIRST ────────────────────────
  //
  // Matches the pattern in initVisualizer(): listen() calls go before any
  // awaited work so events emitted during minimization are never dropped.
  listen<MinProgressPayload>('lb:min-progress', e => {
    onMinProgress(e.payload);
  });

  listen<MinDonePayload>('lb:min-done', e => {
    onMinDone(e.payload);
  });

  listen<string>('lb:min-error', e => {
    setMinimizeBusy(false);
    setStatus('min-status', `✗ ${e.payload}`, 'status-err');
  });

  listen<DockProgressPayload>('lb:dock-progress', e => {
    document.getElementById('dock-progress-text')!.textContent = e.payload.message;
  });

  listen<DockDonePayload>('lb:dock-done', e => {
    onDockDone(e.payload);
  });

  listen<string>('lb:dock-error', e => {
    setDockBusy(false);
    setStatus('dock-status', `✗ ${e.payload}`, 'status-err');
  });

  // ── Step 3: Element grid ───────────────────────────────────────────────────
  const grid = document.getElementById('elem-grid')!;
  for (const el of ORGANIC_ELEMS) {
    const btn = document.createElement('button');
    btn.className   = 'elem-btn' + (el === currentEl ? ' sel' : '');
    btn.textContent = el;
    btn.style.color = ELEMENT_DATA[el]?.color ?? '#aaa';
    btn.addEventListener('click', () => {
      currentEl=el;setTool('atom');
      grid.querySelectorAll('.elem-btn').forEach(b=>b.classList.remove('sel'));
      btn.classList.add('sel');
    });
    grid.appendChild(btn);
  }

  // ── Step 4: Unified pointer events ────────────────────────────────────────

  canvas.addEventListener('contextmenu', e => {
    e.preventDefault();
    const bid=pickBond(e.offsetX,e.offsetY);
    if(bid!==null){saveSnapshot();cycleBondOrder(bid);}
  });

  canvas.addEventListener('mousedown', e => {
    e.preventDefault();
    ptrDown=true;ptrMoved=0;
    ptrStart={x:e.clientX,y:e.clientY};ptrLast={x:e.clientX,y:e.clientY};dragMode='none';
    if(e.button===1||(e.button===0&&e.altKey)){
      dragMode='pan';viewPanStart={x:viewPanX,y:viewPanY};canvas.style.cursor='grabbing';
    } else if(e.button===0){
      if(tool==='bond'){
        const hit=pickAtom(e.offsetX,e.offsetY);
        if(hit!==null){dragMode='bond';bondDragStart=hit;bondPreviewPx={x:e.offsetX,y:e.offsetY};}
        else{dragMode='rotate';_viewRotStart=[...viewRot];}
      } else {
        dragMode='rotate';_viewRotStart=[...viewRot];
      }
    }
  });

  document.addEventListener('mousemove', e => {
    if(!ptrDown){
      const hit=pickAtom(e.offsetX,e.offsetY);
      if(hit!==null&&canvas.matches(':hover')){
        const a=atomById(hit)!;
        tooltip.textContent=`${a.el}  (${a.x.toFixed(2)}, ${a.y.toFixed(2)}, ${a.z.toFixed(2)}) Å`;
        tooltip.style.left=(e.offsetX+14)+'px';tooltip.style.top=(e.offsetY+8)+'px';tooltip.style.opacity='1';
      } else tooltip.style.opacity='0';
      return;
    }
    const dx=e.clientX-ptrLast.x,dy=e.clientY-ptrLast.y;
    ptrMoved+=Math.hypot(e.clientX-ptrStart.x,e.clientY-ptrStart.y);
    ptrLast={x:e.clientX,y:e.clientY};
    if(dragMode==='rotate'){applyRotationDelta(dx,dy);render3D();}
    else if(dragMode==='pan'){viewPanX=viewPanStart.x+(e.clientX-ptrStart.x);viewPanY=viewPanStart.y+(e.clientY-ptrStart.y);render3D();}
    else if(dragMode==='bond'){bondPreviewPx={x:e.offsetX,y:e.offsetY};bondHoverAtom=pickAtom(e.offsetX,e.offsetY);render3D();}
  });

  document.addEventListener('mouseup', e => {
    if(!ptrDown) return;
    ptrDown=false;
    const wasDrag=ptrMoved>DRAG_THRESHOLD,ox=e.offsetX,oy=e.offsetY;
    if(dragMode==='pan'){canvas.style.cursor='crosshair';}
    else if(dragMode==='bond'){
      if(bondDragStart!==null){const hit=pickAtom(ox,oy);if(hit!==null&&hit!==bondDragStart){saveSnapshot();completeBond(bondDragStart,hit);render();}}
      bondDragStart=null;bondPreviewPx=null;bondHoverAtom=null;render3D();
    } else if(dragMode==='rotate'&&!wasDrag){
      const hit=pickAtom(ox,oy);
      if(tool==='select'){
        if(hit!==null){showAtomProps(hit);render();}
        else{selAtomId=null;document.getElementById('atom-props-panel')!.innerHTML='<span style="color:var(--muted)">No atom selected</span>';render();}
      } else if(tool==='atom'){
        saveSnapshot();
        if(hit!==null){atomById(hit)!.el=currentEl;}
        else{const avgZ=atoms.length?atoms.map(a=>project(a).sz).reduce((s,z)=>s+z,0)/atoms.length:0;const w=unproject(ox,oy,avgZ);atoms.push({id:nextAtomId++,el:currentEl,x:w.x,y:w.y,z:w.z,charge:0,radical:0});}
        render();
      } else if(tool==='erase'){
        if(hit!==null){saveSnapshot();deleteAtom(hit);}
        else{const bid=pickBond(ox,oy);if(bid!==null){saveSnapshot();bonds=bonds.filter(b=>b.id!==bid);render();}}
      } else if(tool==='bond'){
        if(hit!==null){saveSnapshot();const a=atomById(hit)!;const ang=Math.random()*2*Math.PI;const nA:LbAtom={id:nextAtomId++,el:currentEl,x:a.x+Math.cos(ang)*BL3,y:a.y+Math.sin(ang)*BL3,z:a.z,charge:0,radical:0};atoms.push(nA);completeBond(hit,nA.id);render();}
      }
    }
    dragMode='none';
  });

  canvas.addEventListener('wheel', e => {
    e.preventDefault();viewZoom*=e.deltaY<0?1.1:0.9;viewZoom=Math.max(10,Math.min(300,viewZoom));render3D();
  },{passive:false});

  // ── Step 5: Toolbar buttons ────────────────────────────────────────────────

  document.getElementById('tool-select')!.addEventListener('click',()=>setTool('select'));
  document.getElementById('tool-atom')!  .addEventListener('click',()=>setTool('atom'));
  document.getElementById('tool-bond')!  .addEventListener('click',()=>setTool('bond'));
  document.getElementById('tool-erase')! .addEventListener('click',()=>setTool('erase'));

  (['1','2','3','a'] as const).forEach(o=>{
    document.getElementById(`bord-${o}`)!.addEventListener('click',()=>{
      bondOrder=o==='a'?'a':parseInt(o,10) as 1|2|3;
      (['1','2','3','a'] as const).forEach(x=>document.getElementById(`bord-${x}`)!.classList.toggle('active',x===o));
      setTool('bond');
    });
  });

  document.getElementById('btn-undo')!          .addEventListener('click',undo);
  document.getElementById('btn-redo')!          .addEventListener('click',redo);
  document.getElementById('btn-minimize-quick')!.addEventListener('click',()=>runMinimize(true));

  document.getElementById('btn-addH')!.addEventListener('click',()=>{
    saveSnapshot();
    const toAdd:{atom:LbAtom;bond:LbBond}[]=[];
    for(const a of atoms.filter(x=>x.el!=='H')){const hc=implicitH(a);for(let k=0;k<hc;k++){const ang=Math.random()*2*Math.PI;const na:LbAtom={id:nextAtomId++,el:'H',x:a.x+Math.cos(ang)*BL3,y:a.y+Math.sin(ang)*BL3,z:a.z+(Math.random()-0.5)*BL3*0.6,charge:0,radical:0};toAdd.push({atom:na,bond:{id:nextBondId++,a1:a.id,a2:na.id,order:1,aromatic:false}});}}
    for(const{atom,bond}of toAdd){atoms.push(atom);bonds.push(bond);}render();
  });

  document.getElementById('btn-clearH')!.addEventListener('click',()=>{
    saveSnapshot();
    const hIds=new Set(atoms.filter(a=>a.el==='H'&&bondsOf(a.id).length<=1).map(a=>a.id));
    atoms=atoms.filter(a=>!hIds.has(a.id));bonds=bonds.filter(b=>!hIds.has(b.a1)&&!hIds.has(b.a2));render();
  });

  document.getElementById('btn-clear')!.addEventListener('click',()=>{
    if(!atoms.length||confirm('Clear all atoms?')){saveSnapshot();atoms=[];bonds=[];selAtomId=null;render();}
  });

  document.getElementById('btn-smiles-load')!.addEventListener('click',()=>{
    const s=(document.getElementById('smiles-in') as HTMLInputElement).value.trim();if(!s) return;
    const parsed=parseSMILES(s);if(!parsed.atoms.length){alert('SMILES parse failed.');return;}
    saveSnapshot();atoms=parsed.atoms;bonds=parsed.bonds;
    nextAtomId=Math.max(...atoms.map(a=>a.id))+1;nextBondId=bonds.length?Math.max(...bonds.map(b=>b.id))+1:1;
    viewRot=[1,0,0,0,1,0,0,0,1];viewZoom=60;viewPanX=0;viewPanY=0;render();
  });
  document.getElementById('smiles-in')!.addEventListener('keydown',e=>{if(e.key==='Enter')document.getElementById('btn-smiles-load')!.click();});
  document.getElementById('btn-smiles-copy')!.addEventListener('click',()=>navigator.clipboard.writeText(toSmiles()).catch(console.error));

  document.querySelectorAll<HTMLButtonElement>('[data-ring]').forEach(btn=>btn.addEventListener('click',()=>insertRing(btn.dataset['ring']!)));

  document.querySelectorAll<HTMLButtonElement>('[data-frag]').forEach(btn=>btn.addEventListener('click',()=>{
    const s=FRAGS[btn.dataset['frag']!];if(!s) return;
    const parsed=parseSMILES(s);if(!parsed.atoms.length) return;
    saveSnapshot();
    const ox=atoms.length?Math.max(...atoms.map(a=>a.x))+BL3*2:0;
    const idMap:Record<number,number>={};
    for(const a of parsed.atoms){const nid=nextAtomId++;idMap[a.id]=nid;atoms.push({...a,id:nid,x:a.x+ox,y:a.y,z:0});}
    for(const b of parsed.bonds) bonds.push({...b,id:nextBondId++,a1:idMap[b.a1],a2:idMap[b.a2]});
    render();
  }));

  document.addEventListener('keydown',e=>{
    if((e.target as HTMLElement).tagName==='INPUT'||(e.target as HTMLElement).tagName==='TEXTAREA') return;
    if(e.ctrlKey&&e.key==='z'){undo();return;}if(e.ctrlKey&&e.key==='y'){redo();return;}
    if(e.key==='v'||e.key==='V') setTool('select');if(e.key==='a'||e.key==='A') setTool('atom');
    if(e.key==='b'||e.key==='B') setTool('bond');  if(e.key==='e'||e.key==='E') setTool('erase');
    if((e.key==='Delete'||e.key==='Backspace')&&selAtomId!==null){saveSnapshot();deleteAtom(selAtomId);}
    if(e.key==='Escape'){selAtomId=null;bondDragStart=null;bondPreviewPx=null;render();}
  });

  // ── Step 6: Right-panel controls ──────────────────────────────────────────

  // Tab switching
  document.querySelectorAll<HTMLButtonElement>('.rtab').forEach(tab=>{
    tab.addEventListener('click',()=>{
      document.querySelectorAll('.rtab').forEach(t=>t.classList.remove('active'));tab.classList.add('active');
      (['props','qm','dock','export'] as RightTab[]).forEach(id=>{(document.getElementById(`tab-${id}`)!).style.display=tab.dataset['tab']===id?'block':'none';});
    });
  });

  // Method picker: show/hide option panels
  document.getElementById('min-method')!.addEventListener('change',()=>{
    const v=(document.getElementById('min-method') as HTMLSelectElement).value;
    document.getElementById('opts-uff')!.style.display  = v==='uff'?'block':'none';
    document.getElementById('opts-xtb')!.style.display  = v.startsWith('gfn')?'block':'none';
    document.getElementById('opts-orca')!.style.display = v.startsWith('orca')?'block':'none';
  });

  // QM custom fields
  document.getElementById('qm-method')!.addEventListener('change',()=>{document.getElementById('method-custom-row')!.style.display=(document.getElementById('qm-method') as HTMLSelectElement).value==='custom'?'flex':'none';});
  document.getElementById('qm-basis')! .addEventListener('change',()=>{document.getElementById('basis-custom-row')! .style.display=(document.getElementById('qm-basis')  as HTMLSelectElement).value==='custom'?'flex':'none';});

  document.getElementById('btn-minimize-full')!  .addEventListener('click',()=>runMinimize(false));
  document.getElementById('btn-minimize-cancel')!.addEventListener('click',cancelMinimize);
  document.getElementById('btn-gen-qm')!         .addEventListener('click',generateQmInput);
  document.getElementById('btn-copy-qm')!        .addEventListener('click',()=>navigator.clipboard.writeText((document.getElementById('qm-preview') as HTMLTextAreaElement).value).catch(console.error));
  document.getElementById('btn-save-qm')!        .addEventListener('click',saveQmInput);

  document.getElementById('btn-save-xyz')!.addEventListener('click',async()=>{try{const s=await invoke<string>('lb_export_xyz',{mol:serializeMol(),coords:serializeCoords()});await saveFile('molecule.xyz','xyz',s,'exp-status');}catch(e){setStatus('exp-status',`✗ ${e}`,'status-err');}});
  document.getElementById('btn-save-pdb')!.addEventListener('click',async()=>{try{const rn=(document.getElementById('exp-resname') as HTMLInputElement).value||'LIG';const s=await invoke<string>('lb_export_pdb',{mol:serializeMol(),coords:serializeCoords(),resname:rn});await saveFile('molecule.pdb','pdb',s,'exp-status');}catch(e){setStatus('exp-status',`✗ ${e}`,'status-err');}});
  document.getElementById('btn-save-mol')!.addEventListener('click',async()=>{try{const s=await invoke<string>('lb_export_mol',{mol:serializeMol(),coords:serializeCoords()});await saveFile('molecule.mol','mol',s,'exp-status');}catch(e){setStatus('exp-status',`✗ ${e}`,'status-err');}});
  document.getElementById('btn-save-sdf')!.addEventListener('click',async()=>{try{const s=await invoke<string>('lb_export_mol',{mol:serializeMol(),coords:serializeCoords()});await saveFile('molecule.sdf','sdf',s+'$$$$\n','exp-status');}catch(e){setStatus('exp-status',`✗ ${e}`,'status-err');}});
  document.getElementById('btn-copy-xyz')!.addEventListener('click',async()=>{try{const s=await invoke<string>('lb_export_xyz',{mol:serializeMol(),coords:serializeCoords()});navigator.clipboard.writeText(s).catch(console.error);}catch(e){console.error(e);}});
  document.getElementById('btn-copy-smiles')!.addEventListener('click',()=>navigator.clipboard.writeText(toSmiles()).catch(console.error));
  document.getElementById('btn-append-qm-region')!.addEventListener('click',async()=>{const rn=(document.getElementById('exp-resname') as HTMLInputElement).value||'LIG';try{const msg=await invoke<string>('lb_append_to_qm_region',{mol:serializeMol(),coords:serializeCoords(),resname:rn});setStatus('exp-status',`✓ ${msg}`,'status-ok');}catch(e){setStatus('exp-status',`✗ ${e}`,'status-err');}});

  // ── Docking controls ──────────────────────────────────────────────────────

  // Box mode radio buttons — three modes
  document.querySelectorAll<HTMLButtonElement>('#dock-box-mode .radio-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      document.querySelectorAll('#dock-box-mode .radio-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const mode = btn.dataset['box'];
      if (mode === 'receptor') await fetchReceptorCenter();
      if (mode === 'ligand')   updateLigandCenter();
      // manual: user edits the fields directly — no update
    });
  });

  // Auto-fetch receptor center when the receptor path field loses focus
  document.getElementById('dock-receptor-path')!.addEventListener('change', async () => {
    if (activeBoxMode() === 'receptor') await fetchReceptorCenter();
  });
  // Also fetch when file is chosen via the browser button (value set programmatically)
  // We trigger this after setting the value in the picker listener below.

  // Receptor file browser
  document.getElementById('btn-pick-receptor')!.addEventListener('click', async () => {
    const { open } = await import('@tauri-apps/plugin-dialog');
    const path = await open({ filters: [{ name: 'Receptor (PDB / PDBQT)', extensions: ['pdb','pdbqt','ent'] }] });
    if (path && typeof path === 'string') {
      (document.getElementById('dock-receptor-path') as HTMLInputElement).value = path;
      // Auto-fetch center if receptor mode is active
      if (activeBoxMode() === 'receptor') await fetchReceptorCenter();
    }
  });

  // Preview box in 3D visualizer
  document.getElementById('btn-preview-box')!.addEventListener('click', openDockPreview);

  // Run docking
  document.getElementById('btn-run-dock')!.addEventListener('click', runDock);
  document.getElementById('btn-cancel-dock')!.addEventListener('click', cancelDock);

  // View docked complex in 3D
  document.getElementById('btn-view-complex')!.addEventListener('click', openDockedComplex);

  // Load selected pose into 3D editor
  document.getElementById('btn-load-pose')!.addEventListener('click', () => {
    const pose = dockPoses[dockSelectedPose];
    if (!pose || pose.coords.length !== atoms.length) {
      setStatus('dock-status', '✗ Pose atom count does not match current molecule.', 'status-err');
      return;
    }
    saveSnapshot();
    pose.coords.forEach((c, i) => { atoms[i].x = c.x; atoms[i].y = c.y; atoms[i].z = c.z; });
    render();
    setStatus('dock-status', `✓ Pose ${pose.rank} loaded into editor.`, 'status-ok');
  });

  // Save selected pose as XYZ
  document.getElementById('btn-save-pose-xyz')!.addEventListener('click', async () => {
    const pose = dockPoses[dockSelectedPose];
    if (!pose) return;
    try {
      const xyz = await invoke<string>('lb_export_xyz', { mol: serializeMol(), coords: pose.coords });
      await saveFile(`pose_${pose.rank}.xyz`, 'xyz', xyz, 'dock-status');
    } catch(e) { setStatus('dock-status', `✗ ${e}`, 'status-err'); }
  });

  // Save all poses PDBQT
  document.getElementById('btn-save-all-poses')!.addEventListener('click', async () => {
    if (!dockOutputPdbqt) return;
    try {
      const content = await invoke<string>('lb_read_text', { path: dockOutputPdbqt });
      await saveFile('poses_out.pdbqt', 'pdbqt', content, 'dock-status');
    } catch(e) { setStatus('dock-status', `✗ ${e}`, 'status-err'); }
  });

  window.addEventListener('resize',render);

  // ── Step 7: Initial render ─────────────────────────────────────────────────
  render();
}

window.addEventListener('DOMContentLoaded', initLigandBuilder);

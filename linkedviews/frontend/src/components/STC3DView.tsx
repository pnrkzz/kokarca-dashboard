import { useMemo, useState } from "react";
import MapGL, { Layer, NavigationControl, Popup, Source } from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
import { useLinkedStore } from "../store/useLinkedStore";
import type { STCCell, STCMeta } from "../types/stc";

type Props={geojson:any;cells:STCCell[];meta:STCMeta;metric?:string;metricLabel?:string;unit?:string};
type Hover={longitude:number;latitude:number;province:string;year:number;count:number}|null;
const norm=(v:unknown)=>String(v??"").toLocaleLowerCase("tr-TR").replace(/ı/g,"i").normalize("NFD").replace(/[\u0300-\u036f]/g,"").replace(/[^a-z0-9]/g,"");
const fmt=(n:number)=>new Intl.NumberFormat("tr-TR",{maximumFractionDigits:2}).format(n);
export default function STC3DView({geojson,cells,meta,metric="event_count",metricLabel="Gösterge",unit=""}:Props){
 const selectedProvince=useLinkedStore(s=>s.selectedProvince);
 const selectedYear=useLinkedStore(s=>s.selectedYear);
 const setSelectedProvince=useLinkedStore(s=>s.setSelectedProvince);
 const focusYear=useLinkedStore(s=>s.focusYear);
 const [hover,setHover]=useState<Hover>(null);
 const anomaly=metric.includes("anomaly");
 const max=Math.max(1,...cells.map(c=>Math.abs(c.event_count)));
 const cube=useMemo(()=>{
   const geometries=new Map<string,any>();
   for(const f of geojson?.features??[]){const name=f.properties?.province_name??f.properties?.name??f.properties?.NAME_1??f.properties?.shapeName;if(name)geometries.set(norm(name),f.geometry);}
   const features=cells.filter(c=>Number.isFinite(c.event_count)&&(anomaly||c.event_count>0)).flatMap(c=>{
     const geometry=geometries.get(norm(c.province));if(!geometry)return [];
     const index=c.year-meta.min_year;
     return [{type:"Feature",geometry,properties:{province:c.province,year:c.year,event_count:c.event_count,magnitude:Math.abs(c.event_count),base_height:index*24000,top_height:index*24000+17000}}];
   });
   return {type:"FeatureCollection" as const,features};
 },[geojson,cells,meta.min_year,anomaly]);
 const color:any=anomaly?["case",["<",["get","event_count"],0],["interpolate",["linear"],["get","magnitude"],0,"#cbd5e1",max,"#2563eb"],[">",["get","event_count"],0],["interpolate",["linear"],["get","magnitude"],0,"#cbd5e1",max,"#dc2626"],"#cbd5e1"]:["interpolate",["linear"],["get","magnitude"],0,"#818cf8",Math.max(1,max*.5),"#a78bfa",max,"#f97352"];
 const extrusion:any={id:"stc-3d-cells",type:"fill-extrusion",paint:{"fill-extrusion-base":["get","base_height"],"fill-extrusion-height":["get","top_height"],"fill-extrusion-color":color,"fill-extrusion-opacity":.83,"fill-extrusion-vertical-gradient":true}};
 const selected:any={id:"stc-3d-selected",type:"line",filter:["all",["==",["get","province"],selectedProvince||"__none__"],["==",["get","year"],selectedYear??-1]],paint:{"line-color":"#fff","line-width":3}};
 return <section className="stc-3d-panel"><div className="stc-panel-title"><div><strong>İl geometrili 3B STC · {metricLabel}</strong><div style={{fontSize:12}}>İl alanı = coğrafi konum · Katman yüksekliği = yıl · Renk = gösterge değeri</div></div><div className="stc-3d-hint">Döndür: sağ sürükle · Yakınlaştır: tekerlek · Seç: tıkla</div></div>
  <div className="stc-3d-map">
   <MapGL initialViewState={{longitude:35.5,latitude:40.2,zoom:4.9,pitch:58,bearing:-24}} mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json" interactiveLayerIds={["stc-3d-cells"]} onClick={e=>{const f=e.features?.find(f=>f.layer?.id==="stc-3d-cells");if(f){const p=String(f.properties?.province||"");const y=Number(f.properties?.year);if(p)setSelectedProvince(p);if(Number.isFinite(y))focusYear(y);}}} onMouseMove={e=>{const f=e.features?.find(f=>f.layer?.id==="stc-3d-cells");setHover(f?{longitude:e.lngLat.lng,latitude:e.lngLat.lat,province:String(f.properties?.province||""),year:Number(f.properties?.year),count:Number(f.properties?.event_count)}:null);}} onMouseLeave={()=>setHover(null)}>
    <NavigationControl position="top-right"/><Source id="stc-3d-source" type="geojson" data={cube as any}><Layer {...extrusion}/><Layer {...selected}/></Source>
    {hover&&<Popup longitude={hover.longitude} latitude={hover.latitude} closeButton={false} closeOnClick={false} anchor="bottom"><div style={{color:"#172033",display:"grid",gap:3}}><strong>{hover.province} · {hover.year}</strong><span>{metricLabel}: {fmt(hover.count)} {unit}</span><small>İl ve yılı seçmek için tıklayın</small></div></Popup>}
   </MapGL>
   <div className="stc-3d-axis" aria-hidden="true"><span>{meta.max_year}</span><div className="stc-3d-axis-line"/><span>{meta.min_year}</span><b>ZAMAN</b></div>
   <div className="stc-3d-legend">{anomaly?<><span style={{color:"#93c5fd"}}>Negatif</span><span style={{color:"#cbd5e1"}}>0</span><span style={{color:"#fca5a5"}}>Pozitif</span></>:<><span>Düşük</span><div className="stc-3d-gradient"/><span>Yüksek</span></>}</div>
  </div>
  <p style={{fontSize:12,marginTop:8}}>Harita altlığı yalnızca coğrafi referanstır; 3B katmanlar zamana göre yükseltilmiş şematik gösterimdir. Yükseklik, olay sayısını veya anomali büyüklüğünü göstermez.</p>
 </section>;
}

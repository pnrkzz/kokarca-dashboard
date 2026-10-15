import { useEffect, useMemo, useState } from "react";
import "./App.css";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import ControlBar from "./components/ControlBar";
import MapView from "./components/MapView";
import LinkedTimeline from "./components/LinkedTimeline";
import ContextPanel from "./components/ContextPanel";
import PhasePlaceholder from "./components/PhasePlaceholder";
import DocumentsView from "./components/DocumentsView";
import ClimateView from "./components/ClimateView";
import QuestionsView from "./components/QuestionsView";
import STCView from "./components/STCView";
import MethodologyView from "./components/MethodologyView";
import InterventionView from "./components/InterventionView";
import {
  getAnimation, getAnimationMeta, getBaseGeoJSON, getClimateTimeline,
  getInterventionProvinceDetail, getInterventionTimeline, getMetadata,
  getProvinceDetail, getProvinceYearSummary, getTimeline,
} from "./services/api";
import { useLinkedStore } from "./store/useLinkedStore";
import type {
  AnimationMeta, AnimationResponse, InterventionProvinceDetail, Metadata,
  ProvinceDetail, ProvinceYearSummary, TimelinePoint,
} from "./types";

export default function App() {
  const activeView=useLinkedStore(s=>s.activeView);
  const selectedProvince=useLinkedStore(s=>s.selectedProvince);
  const selectedProduct=useLinkedStore(s=>s.selectedProduct);
  const timeMode=useLinkedStore(s=>s.timeMode);
  const animationKind=useLinkedStore(s=>s.animationKind);
  const selectedYear=useLinkedStore(s=>s.selectedYear);
  const rangeStart=useLinkedStore(s=>s.rangeStart);
  const rangeEnd=useLinkedStore(s=>s.rangeEnd);
  const isPlaying=useLinkedStore(s=>s.isPlaying);
  const setSelectedProvince=useLinkedStore(s=>s.setSelectedProvince);
  const setSelectedYear=useLinkedStore(s=>s.setSelectedYear);
  const setRangeStart=useLinkedStore(s=>s.setRangeStart);
  const setRangeEnd=useLinkedStore(s=>s.setRangeEnd);
  const setIsPlaying=useLinkedStore(s=>s.setIsPlaying);

  const [metadata,setMetadata]=useState<Metadata|null>(null);
  const [baseGeoJSON,setBaseGeoJSON]=useState<any>(null);
  const [provinceYearData,setProvinceYearData]=useState<ProvinceYearSummary[]>([]);
  const [eventTimeline,setEventTimeline]=useState<TimelinePoint[]>([]);
  const [publishedTimeline,setPublishedTimeline]=useState<TimelinePoint[]>([]);
  const [interventionTimeline,setInterventionTimeline]=useState<TimelinePoint[]>([]);
  const [climateTimeline,setClimateTimeline]=useState<TimelinePoint[]>([]);
  const [provinceDetail,setProvinceDetail]=useState<ProvinceDetail|null>(null);
  const [interventionDetail,setInterventionDetail]=useState<InterventionProvinceDetail|null>(null);
  const [animationMeta,setAnimationMeta]=useState<AnimationMeta|null>(null);
  const [animationData,setAnimationData]=useState<AnimationResponse|null>(null);
  const [contextLoading,setContextLoading]=useState(false);

  const firstYear=metadata?.years[0]??2017;
  const lastYear=metadata?.years[(metadata?.years.length??1)-1]??2026;

  useEffect(()=>{
    Promise.all([getMetadata(),getBaseGeoJSON(),getAnimationMeta().catch(() => null)])
      .then(([meta,geo,anim])=>{
        setMetadata(meta); setBaseGeoJSON(geo); setAnimationMeta(anim);
        const first=meta.years[0], last=meta.years[meta.years.length-1];
        setRangeStart(first); setRangeEnd(last); setSelectedYear(last);
      }).catch(console.error);
  },[setRangeStart,setRangeEnd,setSelectedYear]);

  const activeRange=useMemo(()=>{
    if(timeMode==="single") return {startYear:selectedYear,endYear:selectedYear};
    if(timeMode==="range") return {startYear:Math.min(rangeStart,rangeEnd),endYear:Math.max(rangeStart,rangeEnd)};
    if(timeMode==="cumulative") return {startYear:firstYear,endYear:selectedYear};
    if(timeMode==="animation") return animationKind==="annual"
      ? {startYear:selectedYear,endYear:selectedYear}
      : {startYear:firstYear,endYear:selectedYear};
    return {startYear:firstYear,endYear:lastYear};
  },[timeMode,selectedYear,rangeStart,rangeEnd,firstYear,lastYear,animationKind]);

  useEffect(()=>{
    if(!metadata)return;
    getProvinceYearSummary(selectedProduct).then(setProvinceYearData).catch(console.error);
  },[metadata,selectedProduct]);

  useEffect(()=>{
    if(!metadata)return;
    Promise.all([
      getTimeline({product:selectedProduct,province:selectedProvince||undefined,timeBasis:"event"}),
      getTimeline({product:selectedProduct,province:selectedProvince||undefined,timeBasis:"published"}),
      getInterventionTimeline({product:selectedProduct,province:selectedProvince||undefined}),
      getClimateTimeline({province:selectedProvince||undefined,timeBasis:"published"}),
    ]).then(([a,b,c,d])=>{
      setEventTimeline(a); setPublishedTimeline(b); setInterventionTimeline(c); setClimateTimeline(d);
    }).catch(console.error);
  },[metadata,selectedProduct,selectedProvince]);

  useEffect(()=>{
    if(!selectedProvince){
      setProvinceDetail(null); setInterventionDetail(null); setContextLoading(false); return;
    }
    setContextLoading(true);
    Promise.all([
      getProvinceDetail(selectedProvince,{product:selectedProduct,startYear:activeRange.startYear,endYear:activeRange.endYear}),
      getInterventionProvinceDetail(selectedProvince,{product:selectedProduct,startYear:activeRange.startYear,endYear:activeRange.endYear}),
    ]).then(([a,b])=>{setProvinceDetail(a);setInterventionDetail(b);})
      .catch(console.error).finally(()=>setContextLoading(false));
  },[selectedProvince,selectedProduct,activeRange.startYear,activeRange.endYear]);

  useEffect(()=>{
    if(timeMode!=="animation"){setAnimationData(null);return;}
    if(!animationMeta)return;
    const y=Math.min(animationMeta.max_year,Math.max(animationMeta.min_year,selectedYear));
    if(y!==selectedYear){setSelectedYear(y);return;}
    getAnimation({year:y,mode:animationKind,province:selectedProvince||undefined})
      .then(setAnimationData).catch(e=>{console.error("Animation load failed:",e);setAnimationData(null);});
  },[timeMode,animationMeta,selectedYear,animationKind,selectedProvince,setSelectedYear]);

  const availableYears=useMemo(()=>{
    if(timeMode==="animation"&&animationMeta)
      return Object.keys(animationMeta.year_counts).map(Number).filter(Number.isFinite).sort((a,b)=>a-b);
    return Array.from(new Set(provinceYearData.map(x=>x.year))).sort((a,b)=>a-b);
  },[provinceYearData,timeMode,animationMeta]);

  useEffect(()=>{
    if(!isPlaying||timeMode!=="animation"||!availableYears.length)return;
    const timer=window.setInterval(()=>{
      const i=availableYears.indexOf(selectedYear);
      if(i===-1||i+1>=availableYears.length){
        setIsPlaying(false);setSelectedYear(availableYears[availableYears.length-1]);return;
      }
      setSelectedYear(availableYears[i+1]);
    },1400);
    return()=>window.clearInterval(timer);
  },[isPlaying,timeMode,availableYears,selectedYear,setIsPlaying,setSelectedYear]);

  const countMap=useMemo(()=>{
    const m=new Map<string,number>();
    for(const r of provinceYearData){
      if(r.year<activeRange.startYear||r.year>activeRange.endYear)continue;
      m.set(r.province,(m.get(r.province)||0)+r.event_count);
    }
    return m;
  },[provinceYearData,activeRange.startYear,activeRange.endYear]);

  const displayGeoJSON=useMemo(()=>{
    if(!baseGeoJSON)return null;
    return {...baseGeoJSON,features:baseGeoJSON.features.map((f:any)=>{
      const p=String(f.properties?.province_name||"");
      return {...f,properties:{...f.properties,event_count:countMap.get(p)||0}};
    })};
  },[baseGeoJSON,countMap]);

  const maxCount=Math.max(...Array.from(countMap.values()),1);
  const controls=["map","documents","climate","interventions"].includes(activeView);

  return <div className="platform-shell">
    <Sidebar/>
    <div className="platform-main">
      <TopBar metadata={metadata}/>
      {controls&&(activeView==="interventions" ? <div className="intervention-controlbar"><ControlBar metadata={metadata}/></div> : <ControlBar metadata={metadata}/>)}
      {activeView==="map"?<>
        <div className="main-map-row">
          <section className="map-workspace">
            <MapView geojson={displayGeoJSON} selectedProvince={selectedProvince}
              maxCount={maxCount} onProvinceClick={setSelectedProvince}
              animationActive={timeMode==="animation"} animationFeatures={animationData?.features??[]}/>
          </section>
          <ContextPanel provinceDetail={provinceDetail} interventionDetail={interventionDetail}
            eventTimeline={eventTimeline} publishedTimeline={publishedTimeline}
            climateTimeline={climateTimeline} loading={contextLoading}/>
        </div>
        <div className="full-width-time-row">
          <LinkedTimeline metadata={metadata} eventData={eventTimeline}
            publishedData={publishedTimeline} interventionData={interventionTimeline}
            climateData={climateTimeline}/>
        </div>
      </>:activeView==="stc"?
        <STCView/>:
      activeView==="documents"?
        <DocumentsView startYear={activeRange.startYear} endYear={activeRange.endYear}/>:
      activeView==="climate"?
        <ClimateView startYear={activeRange.startYear} endYear={activeRange.endYear}/>:
      activeView==="interventions"?
        <InterventionView detail={interventionDetail} timeline={interventionTimeline}
          startYear={activeRange.startYear} endYear={activeRange.endYear} loading={contextLoading}/>:
      activeView==="questions"?<QuestionsView/>:activeView==="methodology"?<MethodologyView/>:<PhasePlaceholder/>}
    </div>
  </div>;
}




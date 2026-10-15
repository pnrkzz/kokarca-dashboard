import { useMemo, useState } from "react";
import Map, { Layer, NavigationControl, Popup, Source } from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";

import type { AnimationFeature } from "../types";

type Props = {
  geojson: any;
  selectedProvince: string | null;
  maxCount: number;
  onProvinceClick: (province: string) => void;
  animationActive?: boolean;
  animationFeatures?: AnimationFeature[];
  metricLabel?: string;
};

type HoverInfo = {
  longitude: number;
  latitude: number;
  province: string;
  count: number;
} | null;

type AnimationHover = {
  longitude: number;
  latitude: number;
  item: AnimationFeature;
} | null;

export default function MapView({
  geojson,
  selectedProvince,
  maxCount,
  onProvinceClick,
  animationActive = false,
  animationFeatures = [],
  metricLabel = "olay",
}: Props) {
  const [hover, setHover] = useState<HoverInfo>(null);
  const [animationHover, setAnimationHover] =
    useState<AnimationHover>(null);

  const animationGeoJSON = useMemo(
    () => ({
      type: "FeatureCollection" as const,
      features: animationFeatures.map((item) => ({
        type: "Feature" as const,
        geometry: item.geometry,
        properties: {
          animation_uid: item.animation_uid,
          animation_year: item.animation_year,
          animation_type: item.animation_type,
          display_location: item.display_location,
          province: item.province,
          district: item.district,
          geometry_precision: item.geometry_precision,
        },
      })),
    }),
    [animationFeatures],
  );

  const fillLayer: any = {
    id: "province-fill",
    type: "fill",
    paint: {
      "fill-color": [
        "interpolate",
        ["linear"],
        ["get", "event_count"],
        0, "#171a2e",
        Math.max(1, maxCount) * 0.12, "#333460",
        Math.max(1, maxCount) * 0.35, "#6d4ec5",
        Math.max(1, maxCount) * 0.65, "#b44f8d",
        Math.max(1, maxCount), "#ef7d64",
      ],
      "fill-opacity": animationActive ? 0.30 : 0.84,
    },
  };

  const lineLayer: any = {
    id: "province-line",
    type: "line",
    paint: {
      "line-color": [
        "case",
        ["==", ["get", "province_name"], selectedProvince || ""],
        "#f4b57a",
        "#6b6d94",
      ],
      "line-width": [
        "case",
        ["==", ["get", "province_name"], selectedProvince || ""],
        2.8,
        0.7,
      ],
      "line-opacity": 0.9,
    },
  };

  const animationLayer: any = {
    id: "animation-points",
    type: "circle",
    paint: {
      "circle-radius": [
        "match",
        ["get", "animation_type"],
        "first_detection", 8,
        "spread_destination", 7,
        "direct_observation", 6,
        "research_observation", 5,
        "spread_observation", 5,
        5,
      ],
      "circle-color": [
        "match",
        ["get", "animation_type"],
        "first_detection", "#ffb347",
        "direct_observation", "#57c7ff",
        "spread_destination", "#ff5c8a",
        "research_observation", "#b79cff",
        "spread_observation", "#68d391",
        "#ffffff",
      ],
      "circle-opacity": 0.92,
      "circle-stroke-width": 1.4,
      "circle-stroke-color": "#ffffff",
    },
  };

  return (
    <div className="map-shell">
      <Map
        initialViewState={{
          longitude: 35.2,
          latitude: 39,
          zoom: 5.2,
        }}
        mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
        interactiveLayerIds={
          animationActive
            ? ["animation-points", "province-fill"]
            : ["province-fill"]
        }
        onClick={(event) => {
          const point = event.features?.find(
            (f) => f.layer?.id === "animation-points",
          );

          if (point) return;

          const feature = event.features?.find(
            (f) => f.layer?.id === "province-fill",
          );

          if (!feature) return;

          const province = String(
            feature.properties?.province_name || "",
          );

          if (province) onProvinceClick(province);
        }}
        onMouseMove={(event) => {
          const point = event.features?.find(
            (f) => f.layer?.id === "animation-points",
          );

          if (animationActive && point) {
            const uid = String(
              point.properties?.animation_uid || "",
            );

            const item = animationFeatures.find(
              (x) => x.animation_uid === uid,
            );

            if (item) {
              setHover(null);
              setAnimationHover({
                longitude: event.lngLat.lng,
                latitude: event.lngLat.lat,
                item,
              });
              return;
            }
          }

          setAnimationHover(null);

          const feature = event.features?.find(
            (f) => f.layer?.id === "province-fill",
          );

          if (!feature) {
            setHover(null);
            return;
          }

          setHover({
            longitude: event.lngLat.lng,
            latitude: event.lngLat.lat,
            province: String(
              feature.properties?.province_name || "",
            ),
            count: Number(
              feature.properties?.event_count || 0,
            ),
          });
        }}
        onMouseLeave={() => {
          setHover(null);
          setAnimationHover(null);
        }}
      >
        <NavigationControl position="top-right" />

        {geojson && (
          <Source id="provinces" type="geojson" data={geojson}>
            <Layer {...fillLayer} />
            <Layer {...lineLayer} />
          </Source>
        )}

        {animationActive && (
          <Source
            id="animation-source"
            type="geojson"
            data={animationGeoJSON}
          >
            <Layer {...animationLayer} />
          </Source>
        )}

        {animationHover && (
          <Popup
            longitude={animationHover.longitude}
            latitude={animationHover.latitude}
            closeButton={false}
            closeOnClick={false}
            anchor="bottom"
          >
            <div className="map-tooltip">
              <strong>
                {animationHover.item.display_location}
              </strong>
              <span>
                {animationHover.item.animation_year}
              </span>
              <span>
                {animationHover.item.animation_type}
              </span>
              <span>
                {animationHover.item.geometry_precision}
              </span>
            </div>
          </Popup>
        )}

        {!animationHover && hover && (
          <Popup
            longitude={hover.longitude}
            latitude={hover.latitude}
            closeButton={false}
            closeOnClick={false}
            anchor="bottom"
          >
            <div className="map-tooltip">
              <strong>{hover.province}</strong>
              <span>{hover.count} {metricLabel}</span>
            </div>
          </Popup>
        )}
      </Map>

      <div className="map-legend">
        <span className="legend-title">
          {animationActive
            ? "Zamansal H. halys kayıtları"
            : metricLabel === "olay" ? "Olay yoğunluğu" : "Kaynaklı iklim ilişkileri"}
        </span>

        {animationActive ? (
          <div style={{ fontSize: 11, lineHeight: 1.7 }}>
            <div>Turuncu — İlk tespit</div>
            <div>Mavi — Doğrudan gözlem</div>
            <div>Pembe — Yayılım hedefi</div>
            <div>Mor — Araştırma gözlemi</div>
            <div>Yeşil — Yayılım gözlemi</div>
          </div>
        ) : (
          <>
            <div className="legend-scale">
              <i className="legend-c1" />
              <i className="legend-c2" />
              <i className="legend-c3" />
              <i className="legend-c4" />
              <i className="legend-c5" />
            </div>

            <div className="legend-labels">
              <span>Düşük</span>
              <span>Yüksek</span>
            </div>
          </>
        )}
      </div>

      <div className="map-help">
        {animationActive
          ? "Noktalar idari temsil koordinatlarıdır; kesin gözlem koordinatı değildir."
          : metricLabel === "olay" ? "İl seç → zaman, müdahale ve kaynak bağlamı birlikte güncellensin" : "İl seç → iklim ilişkileri ve kaynak belgeler filtrelensin"}
      </div>
    </div>
  );
}

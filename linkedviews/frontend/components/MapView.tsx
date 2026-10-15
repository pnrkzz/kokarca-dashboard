import Map, {
  Layer,
  NavigationControl,
  Popup,
  Source,
} from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";

type Props = {
  geojson: any;
  selectedProvince: string | null;
  maxCount: number;
  onProvinceClick: (province: string) => void;
  label: string;
};

export default function MapView({
  geojson,
  selectedProvince,
  maxCount,
  onProvinceClick,
  label,
}: Props) {
  const fillLayer: any = {
    id: "province-fill",
    type: "fill",
    paint: {
      "fill-color": [
        "interpolate",
        ["linear"],
        ["get", "event_count"],
        0,
        "#eef3f4",
        Math.max(1, maxCount * 0.2),
        "#9abdb1",
        Math.max(2, maxCount * 0.5),
        "#d7a95c",
        maxCount,
        "#a23d34",
      ],
      "fill-opacity": 0.82,
    },
  };

  const lineLayer: any = {
    id: "province-line",
    type: "line",
    paint: {
      "line-color": "#ffffff",
      "line-width": [
        "case",
        ["==", ["get", "province_name"], selectedProvince || ""],
        3,
        0.7,
      ],
      "line-opacity": 0.85,
    },
  };

  return (
    <Map
      initialViewState={{
        longitude: 35.2,
        latitude: 39,
        zoom: 5.2,
      }}
      mapStyle="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
      interactiveLayerIds={["province-fill"]}
      onClick={(event) => {
        const feature = event.features?.[0];
        if (!feature) return;
        const province = String(feature.properties?.province_name || "");
        if (province) onProvinceClick(province);
      }}
    >
      <NavigationControl position="top-right" />

      {geojson && (
        <Source id="provinces" type="geojson" data={geojson}>
          <Layer {...fillLayer} />
          <Layer {...lineLayer} />
        </Source>
      )}
    </Map>
  );
}

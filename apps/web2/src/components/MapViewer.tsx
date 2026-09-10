import { useEffect, useRef } from 'react';
import Map from 'ol/Map';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import 'ol/ol.css';

export const MapViewer = () => {
  const mapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mapRef.current) return;

    const map = new Map({
      target: mapRef.current,
      layers: [
        new TileLayer({
          source: new OSM(),
        }),
      ],
      view: new View({
        center: [0, 0],
        zoom: 2,
      }),
    });

    return () => {
      map.setTarget(undefined);
    };
  }, []);

  return (
    <div className="absolute inset-0 w-full h-full z-0 hidden" ref={mapRef}>
      {/* Map is hidden by default in this layout until a query activates it */}
    </div>
  );
};

// Control de pestañas
function openTab(evt, tabName) {
    var i;
    var x = document.getElementsByClassName("tabContent");
    var y = document.getElementsByClassName("tab");
    for (i = 0; i < x.length; i++) {
       x[i].style.display = "none";
    }
    for (i = 0; i < y.length; i++) {
       y[i].className = y[i].className.replace(" clicked", "");
    }
    document.getElementById(tabName).style.display = "grid";
    evt.currentTarget.className += " clicked";
    evt.currentTarget.blur();
}

// Mapa
var map = L.map('map', {attributionControl:false}).setView([37.38, -5.98], 12);

// Capa base
L.esri.basemapLayer("Topographic").addTo(map);

// Marcador coloreado según gradiente para capa estación
// por la propiedad stands
function getColor(d) {
    return d >= 40  ? '#FF0000' :
           d >= 35  ? '#FF5000' :
           d >= 30  ? '#FFAA00' :
           d >= 25  ? '#FFD100' :
           d >= 20  ? '#FFF000' :
           d >= 15  ? '#1FFF00' :
           d >= 10  ? '#00FF90 ' :
                      '#00FFF0';
}
function style(feature) {
    return {
        fillColor: getColor(feature.properties.stands),	
		radius: 4,
		color: "#ff0000",
		weight: 1,
		opacity: 1,
		fillOpacity: 0.5
	}
};

// Capa estaciones desde CSV
// Usamos extensión Leaflet geoCSV
var estaciones = L.geoCsv(null, {
	fieldSeparator: ',',
	firstLineTitles: true,
	latitudeTitle: 'latitude',
	longitudeTitle: 'longitude',
	onEachFeature: function (feature, layer) {
		var popup = '';
		for (var clave in feature.properties) {
			var title = estaciones.getPropertyTitle(clave);
			popup += '<b>'+title+'</b>: '+feature.properties[clave]+'<br />';
		}
		layer.bindPopup(popup);
	},
	pointToLayer: function (feature, latlng) {
		return L.circleMarker(latlng, style(feature));
	}
});

// Usamos lib jquery para carga asíncrona del csv
$.ajax ({
	type:'GET',
	dataType:'text',
	url:'data/seviesta.csv',
	error: function() {
		alert('No se pudieron cargar los datos');
	},
	success: function(csv) {
		estaciones.addData(csv);
		map.addLayer(estaciones);
	}
});

// Leyenda para capa estaciones
var legend = L.control({position: 'bottomleft'});

legend.onAdd = function (map) {
    var div = L.DomUtil.create('div', 'legend'),
        grades = [0, 10, 15, 20, 25, 30, 35, 40],
        labels = [];
    div.innerHTML = '<h4>Estacionamientos <br> operativos</h4>'
    for (var i = 0; i < grades.length; i++) {
        div.innerHTML +=
            '<i style="background:' + getColor(grades[i] + 1) + '"></i> ' +
            grades[i] + (grades[i + 1] ? '&ndash;' + grades[i + 1] + '<br>' : '+');
    }
    return div;
};

legend.addTo(map);

// Popup coordenadas al click
var popup = L.popup();

function onMapClick(e) {
	popup
		.setLatLng(e.latlng)
		.setContent("Coordenadas: " + e.latlng.toString())
		.openOn(map);
}

map.on('click', onMapClick);

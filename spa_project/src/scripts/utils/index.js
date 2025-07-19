import Swal from 'sweetalert2';

export function showFormattedDate(date, locale = 'id-ID', options = {}) {
  const defaultOptions = {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false 
  };
  
  return new Date(date).toLocaleDateString(locale, {
    ...defaultOptions,
    ...options,
  });
}

export function sleep(time = 1000) {
  return new Promise((resolve) => setTimeout(resolve, time));
}

export function createSkeletonStory(count = 6) {
  return Array(count).fill(`
    <article class="story-item skeleton">
      <div class="story-item__thumbnail"></div>
      <div class="story-item__content">
        <h3 class="story-item__title"></h3>
        <p class="story-item__description"></p>
        <p class="story-item__date"></p>
      </div>
    </article>
  `).join('');
}

export function showAlert(message, type = 'success') {
  const iconMap = {
    'success': 'success',
    'error': 'error',
    'warning': 'warning',
    'info': 'info'
  };
  
  const icon = iconMap[type] || 'success';
  
  Swal.fire({
    text: message,
    icon: icon,
    toast: true,
    position: 'top-end',
    showConfirmButton: false,
    timer: 3000,
    timerProgressBar: true,
    didOpen: (toast) => {
      toast.addEventListener('mouseenter', Swal.stopTimer);
      toast.addEventListener('mouseleave', Swal.resumeTimer);
    }
  });
}

export function initMap(elementId, lat, lng, popupContent = '') {
  const map = L.map(elementId).setView([lat, lng], 13);
  
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(map);

  const marker = L.marker([lat, lng]).addTo(map);
  if (popupContent) {
    marker.bindPopup(popupContent).openPopup();
  }

  return map;
}

export function initMapWithLayerControl(elementId, lat, lng) {
  const map = L.map(elementId).setView([lat, lng], 13);
  
  const baseLayers = {
    "OpenStreetMap": L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }),
    "Satellite": L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    })
  };

  baseLayers["OpenStreetMap"].addTo(map);
  L.control.layers(baseLayers).addTo(map);

  return map;
}
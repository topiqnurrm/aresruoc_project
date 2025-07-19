import { initMapWithLayerControl, showAlert } from '../utils';

class MapLocationHelper {
  constructor() {
    this._map = null;
    this._marker = null;
    this._userLocation = null;
    
    this._getCurrentPosition = this._getCurrentPosition.bind(this);
    this._getUserLocation = this._getUserLocation.bind(this);
    this._updateSelectedLocation = this._updateSelectedLocation.bind(this);
  }
  
  init(mapElementId, latitudeElementId, longitudeElementId) {
    this.mapElementId = mapElementId;
    this.latitudeElement = document.getElementById(latitudeElementId);
    this.longitudeElement = document.getElementById(longitudeElementId);
    
    this._initMap();
    
    const locationButton = document.getElementById('get-user-location');
    if (locationButton) {
      locationButton.addEventListener('click', this._getUserLocation);
    }
    
    return this;
  }
  
  _initMap() {
    this._map = initMapWithLayerControl(this.mapElementId, -6.1754, 106.8272);
    
    this._map.on('click', (e) => {
      this._updateSelectedLocation(e.latlng.lat, e.latlng.lng);
    });
  }
  
  _updateSelectedLocation(lat, lng) {
    this._userLocation = { lat, lng };
    
    if (this.latitudeElement && this.longitudeElement) {
      this.latitudeElement.textContent = lat.toFixed(6);
      this.longitudeElement.textContent = lng.toFixed(6);
    }
    
    if (this._marker) {
      this._map.removeLayer(this._marker);
    }
    
    this._marker = L.marker([lat, lng]).addTo(this._map)
      .bindPopup('Your story location')
      .openPopup();
  }
  
  async _getUserLocation() {
    try {
      const position = await this._getCurrentPosition();
      const { latitude: lat, longitude: lng } = position.coords;
      
      this._map.setView([lat, lng], 15);
      
      this._updateSelectedLocation(lat, lng);
      
      showAlert('Your location has been set!');
    } catch (error) {
      console.error('Error getting location:', error);
      showAlert(`Could not get your location: ${error.message}`, 'error');
    }
  }
  
  _getCurrentPosition() {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error('Geolocation is not supported by your browser'));
        return;
      }

      navigator.geolocation.getCurrentPosition(
        resolve,
        (error) => {
          let errorMessage = 'Error getting your location';
          switch(error.code) {
            case error.PERMISSION_DENIED:
              errorMessage = 'Location access was denied';
              break;
            case error.POSITION_UNAVAILABLE:
              errorMessage = 'Location information is unavailable';
              break;
            case error.TIMEOUT:
              errorMessage = 'Location request timed out';
              break;
          }
          reject(new Error(errorMessage));
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 0
        }
      );
    });
  }
  
  invalidateSize() {
    if (this._map) {
      this._map.invalidateSize();
    }
  }
  
  get userLocation() {
    return this._userLocation;
  }
}

export default MapLocationHelper;
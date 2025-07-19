import { showAlert } from '../../../utils';
import AddStoryPresenter from '../../../presenters/story/add-story-presenter';
import CameraHelper from '../../../utils/camera-helper';
import MapLocationHelper from '../../../utils/map-location-helper';
import AnimationHelper from '../../../utils/animation-helper';

export default class AddStoryPage {
  constructor() {
    this.cameraHelper = new CameraHelper();
    this.mapLocationHelper = new MapLocationHelper();
    this.presenter = new AddStoryPresenter(this);
  }

  async render() {
    return `
        <div class="container">
          <section class="add-story">
            <form id="story-form" class="story-form">
              <h1 class="add-story__title" style="opacity: 0; transform: translateY(-20px);">Share Your Story</h1>
              <div class="form-group" style="opacity: 0; transform: translateY(20px);">
                <label for="description">Description</label>
                <textarea 
                  id="description" 
                  name="description" 
                  class="form-control" 
                  rows="5" 
                  required
                  aria-required="true"
                  placeholder="Share your story..."
                ></textarea>
              </div>
              
              <div class="form-group" style="opacity: 0; transform: translateY(20px);">
                <label>Photo</label>
                <div class="photo-options">
                  <button type="button" id="open-camera" class="open-camera-button">
                    <i class="fas fa-camera"></i> Open Camera
                  </button>
                  
                  <div class="camera-container">
                    <div class="video-wrapper">
                      <video id="camera-preview" class="camera-preview" autoplay playsinline></video>
                    </div>                  
                    <div id="camera-controls" class="camera-controls">
                      <button type="button" id="take-photo" class="camera-button">
                        <i class="fas fa-camera"></i>  
                      </button>
                    </div>
                  </div>
                  <div class="file-upload">
                    <label for="photo-upload" class="upload-label">Or upload a photo :</label>
                    <input type="file" id="photo-upload" accept="image/*" class="form-control" />
                  </div>
                </div>
                <div id="photo-preview" class="photo-preview"></div>
              </div>
              
              <div class="form-group" style="opacity: 0; transform: translateY(20px);">
                <label>Location</label>
                <div class="location-controls">
                  <button type="button" id="get-user-location" class="location-button">
                    <i class="fas fa-map-marker-alt"></i> Use My Location
                  </button>
                </div>
                <div class="location-hint">
                  <i class="fas fa-info-circle"></i> Or click on map to select a location
                </div>
                <div id="location-map" class="location-map"></div>
                <div class="location-coordinates">
                  <span>Latitude: <span id="latitude">Not selected</span></span>
                  <span>Longitude: <span id="longitude">Not selected</span></span>
                </div>
              </div>
              
              <button type="submit" class="submit-button" style="opacity: 0; transform: translateY(20px);">
                <i class="fas fa-paper-plane"></i>Post Story
              </button>
            </form>
          </section>
        </div>
    `;
  }

  async afterRender() {
    this.cameraHelper.init('camera-container', 'camera-preview', 'camera-controls', 'photo-preview');
    this.mapLocationHelper.init('location-map', 'latitude', 'longitude');
    
    this._setupForm();
    this._setupCameraButton();
    this._animateFormElements();
    
    setTimeout(() => {
      this.mapLocationHelper.invalidateSize();
    }, 100);
    
    window.addEventListener('hashchange', this._cleanup);
  }

  _cleanup = () => {
    this.cameraHelper.cleanup();
    window.removeEventListener('hashchange', this._cleanup);
  }

  _setupCameraButton() {
    const openCameraBtn = document.getElementById('open-camera');
    const fileInput = document.getElementById('photo-upload');
    
    openCameraBtn.addEventListener('click', async () => {
      const cameraContainer = document.querySelector('.camera-container');
      cameraContainer.style.display = 'block';
      openCameraBtn.style.display = 'none'; 
      
      const success = await this.cameraHelper.startCamera();
      if (!success) {
        showAlert('Could not access camera. Please use file upload instead.', 'error');
        cameraContainer.style.display = 'none';
        openCameraBtn.style.display = 'block';
      }
    });
    
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        this.cameraHelper.handleFileUpload(file);
      }
    });
  }

  _animateFormElements() {
    const title = document.querySelector('.add-story__title');
    const formGroups = document.querySelectorAll('.form-group');
    const submitButton = document.querySelector('.submit-button');
    
    AnimationHelper.animateElement(title, {
      delay: 100,
      transform: 'translateY(0)'
    });
    
    AnimationHelper.animateFormElements(formGroups);
    
    AnimationHelper.animateElement(submitButton, {
      delay: 200 + (formGroups.length * 150),
      transform: 'translateY(0)'
    });
  }

  _setupForm() {
    const form = document.getElementById('story-form');
    const submitButton = form.querySelector('.submit-button');
    
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      
      const photoFile = this.cameraHelper.photoFile;
      if (!photoFile) {
        showAlert('Please add a photo', 'error');
        return;
      }
      
      const description = document.getElementById('description').value;
      const userLocation = this.mapLocationHelper.userLocation;
      
      const originalButtonContent = submitButton.innerHTML;
      submitButton.classList.add('loading');
      submitButton.innerHTML = 'Posting...<i class="fas fa-spinner fa-spin"></i>'; 
      submitButton.disabled = true;
      
      try {
        await this.presenter.handleAddStory({
          description,
          photo: photoFile,
          lat: userLocation?.lat,
          lon: userLocation?.lng
        });
        
        this.cameraHelper.cleanup();
      } catch (error) {
        submitButton.classList.remove('loading');
        submitButton.innerHTML = originalButtonContent;
        submitButton.disabled = false;
      }
    });
  }
}
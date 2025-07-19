class CameraHelper {
    constructor() {
      this._mediaStream = null;
      this._photoFile = null;
      this._isCameraActive = false;
      
      this._takePhoto = this._takePhoto.bind(this);
      this._resetPhoto = this._resetPhoto.bind(this);
      this._stopCameraStream = this._stopCameraStream.bind(this);
    }
  
    init(cameraContainerId, previewId, controlsId, photoPreviewId) {
      this.cameraContainer = document.getElementById(cameraContainerId);
      this.videoElement = document.getElementById(previewId);
      this.controlsContainer = document.getElementById(controlsId);
      this.photoPreviewContainer = document.getElementById(photoPreviewId);
      
      const takePhotoBtn = this.controlsContainer.querySelector('#take-photo');
      if (takePhotoBtn) {
        takePhotoBtn.addEventListener('click', this._takePhoto);
      }
      
      window.addEventListener('beforeunload', this._stopCameraStream);
      window.addEventListener('pagehide', this._stopCameraStream);

      return this;
    }
  
    async startCamera() {
      this._stopCameraStream();
  
      try {
        this._mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { 
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 }
          },
          audio: false
        });
        
        this.videoElement.srcObject = this._mediaStream;
        this._isCameraActive = true;
        
        this.videoElement.style.display = 'block';
        this.photoPreviewContainer.style.display = 'none';
        
        return true;
      } catch (error) {
        console.error('Camera error:', error);
        this._isCameraActive = false;
        return false;
      }
    }
  
    _takePhoto() {
      if (!this._isCameraActive) return null;
      
      const canvas = document.createElement('canvas');
      canvas.width = this.videoElement.videoWidth;
      canvas.height = this.videoElement.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height);
      
      canvas.toBlob((blob) => {
        this._photoFile = new File([blob], 'photo.jpg', { type: 'image/jpeg' });
        
        this.photoPreviewContainer.innerHTML = `
          <div class="preview-container">
            <img src="${URL.createObjectURL(blob)}" alt="Captured photo" class="preview-image">
            <button type="button" id="reset-photo" class="reset-photo-button-text">
              <i class="fas fa-times"></i> Remove Photo
            </button>
          </div>
        `;
        
        document.getElementById('reset-photo').addEventListener('click', this._resetPhoto);
        
        this.photoPreviewContainer.style.display = 'block';
        
      }, 'image/jpeg', 0.9);
      
      return this._photoFile;
    }
    
    handleFileUpload(file) {
      this._photoFile = file;
      
      const reader = new FileReader();
      
      reader.onload = (event) => {
        this.photoPreviewContainer.innerHTML = `
          <div class="preview-container">
            <img src="${event.target.result}" alt="Uploaded photo" class="preview-image">
            <button type="button" id="reset-photo" class="reset-photo-button-text">
              <i class="fas fa-times"></i> Remove Photo
            </button>
          </div>
        `;
        
        document.getElementById('reset-photo').addEventListener('click', this._resetPhoto);
        
        this.photoPreviewContainer.style.display = 'block';
      };
      
      reader.readAsDataURL(file);
      
      return this._photoFile;
    }
  
    _resetPhoto() {
      this._photoFile = null;
      this.photoPreviewContainer.innerHTML = '';
      this.photoPreviewContainer.style.display = 'none';
      
      if (this.videoElement) {
        this.videoElement.style.display = 'block';
      }
      
      return null;
    }
  
    _stopCameraStream() {
      if (this._mediaStream) {
        this._mediaStream.getTracks().forEach(track => track.stop());
        if (this.videoElement) {
          this.videoElement.srcObject = null;
        }
        this._mediaStream = null;
        this._isCameraActive = false;
      }
    }
    
    cleanup() {
      window.removeEventListener('beforeunload', this._stopCameraStream);
      window.removeEventListener('pagehide', this._stopCameraStream);
      this._stopCameraStream();
    }
    
    get photoFile() {
      return this._photoFile;
    }
  }
  
  export default CameraHelper;
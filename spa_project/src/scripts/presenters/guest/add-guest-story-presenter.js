import StoryApi from '../../data/api';
import { showAlert } from '../../utils';
import ErrorHandler from '../../utils/error-handler';
import NetworkStatus from '../../utils/network-status';

export default class AddGuestStoryPresenter {
  constructor(view) {
    this._view = view;
    this._storyApi = StoryApi;
  }

  async handleAddStory({ description, photo, lat, lon }) {
    try {
      if (!photo) {
        showAlert('Please add a photo', 'error');
        return;
      }

      if (!description.trim()) {
        showAlert('Please add a description', 'error');
        return;
      }

      const response = await this._storyApi.addGuestStory({
        description,
        photo,
        lat,
        lon
      });

      if (!response.error) {
        showAlert('Guest story added successfully!', 'success');
        
        setTimeout(() => {
          window.location.hash = '#/';
        }, 1500);
        
        return true;
      } else {
        throw new Error(response.message || 'Failed to add guest story');
      }
    } catch (error) {
      if (!NetworkStatus.isOnline) {
        return ErrorHandler.handleOfflineError(error, {
          offlineAction: () => {
            showAlert('Your guest story has been saved offline and will be uploaded when you are back online', 'info');
            
            setTimeout(() => {
              window.location.hash = '#/';
            }, 2000);
            
            return true;
          }
        });
      } else {
        return ErrorHandler.handleFormError(error, {
          context: 'Adding guest story',
          defaultMessage: 'Failed to add guest story. Please try again.'
        });
      }
    }
  }
}
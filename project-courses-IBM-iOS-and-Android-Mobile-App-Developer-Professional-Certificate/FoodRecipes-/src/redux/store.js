// store.js
import { configureStore } from "@reduxjs/toolkit";
import favoritesReducer from "./favoritesSlice"; // Import the favorites reducer

const store = configureStore({
  reducer: {
    favorites: favoritesReducer, // Set the favorites reducer in the store
  },
});

export default store;

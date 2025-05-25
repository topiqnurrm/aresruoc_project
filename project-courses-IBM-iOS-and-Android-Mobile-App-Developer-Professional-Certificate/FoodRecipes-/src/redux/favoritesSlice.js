// favoritesSlice.js
import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  favoriterecipes: [], // This will store the favorite recipes
};

const favoritesSlice = createSlice({
  name: "favorites",
  initialState,
  reducers: {
    toggleFavorite: (state, action) => {
      const recipe = action.payload; // The recipe passed in the action
      const existingRecipeIndex = state.favoriterecipes.findIndex(
        (fav) => fav.idFood === recipe.idFood
      );

      if (existingRecipeIndex !== -1) {
        // If the recipe is already in the favorites, remove it
        state.favoriterecipes.splice(existingRecipeIndex, 1);
      } else {
        // If the recipe is not in the favorites, add it
        state.favoriterecipes.push(recipe);
      }
    },
  },
});

export const { toggleFavorite } = favoritesSlice.actions;
export default favoritesSlice.reducer;

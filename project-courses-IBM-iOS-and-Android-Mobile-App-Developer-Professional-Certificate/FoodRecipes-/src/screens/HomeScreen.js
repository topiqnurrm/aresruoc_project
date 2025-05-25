import React, { useState } from 'react';
import { View, Text, ScrollView, StyleSheet } from 'react-native';
import Categories from './Categories';
import Recipe from './Recipe';

export default function HomePage() {
  const [categories, setCategories] = useState([
    { idCategory: "1", strCategory: "Beef", strCategoryThumb: "https://www.themealdb.com/images/category/beef.png" },
    { idCategory: "2", strCategory: "Chicken", strCategoryThumb: "https://www.themealdb.com/images/category/chicken.png" },
    { idCategory: "3", strCategory: "Dessert", strCategoryThumb: "https://www.themealdb.com/images/category/dessert.png" },
    { idCategory: "4", strCategory: "Lamb", strCategoryThumb: "https://www.themealdb.com/images/category/lamb.png" },
    { idCategory: "5", strCategory: "Miscellaneous", strCategoryThumb: "https://www.themealdb.com/images/category/miscellaneous.png" },
    { idCategory: "6", strCategory: "Pasta", strCategoryThumb: "https://www.themealdb.com/images/category/pasta.png" },
    { idCategory: "7", strCategory: "Pork", strCategoryThumb: "https://www.themealdb.com/images/category/pork.png" },
    { idCategory: "8", strCategory: "Seafood", strCategoryThumb: "https://www.themealdb.com/images/category/seafood.png" },
    { idCategory: "9", strCategory: "Side", strCategoryThumb: "https://www.themealdb.com/images/category/side.png" },
    { idCategory: "10", strCategory: "Starter", strCategoryThumb: "https://www.themealdb.com/images/category/starter.png" },
    { idCategory: "11", strCategory: "Vegan", strCategoryThumb: "https://www.themealdb.com/images/category/vegan.png" },
    { idCategory: "12", strCategory: "Vegetarian", strCategoryThumb: "https://www.themealdb.com/images/category/vegetarian.png" },
    { idCategory: "13", strCategory: "Breakfast", strCategoryThumb: "https://www.themealdb.com/images/category/breakfast.png" },
    { idCategory: "14", strCategory: "Goat", strCategoryThumb: "https://images.unsplash.com/photo-1619711667542-c049700dd9e0?q=80&w=1888&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" },
  ]);

  const [allFood, setAllFood] = useState([
    { idMeal: "1", strMeal: "Beef Stew", strCategory: "Beef", strMealThumb: "https://www.themealdb.com/images/media/meals/uwpsvq1487340131.jpg" },
    { idMeal: "2", strMeal: "Chicken Curry", strCategory: "Chicken", strMealThumb: "https://www.themealdb.com/images/media/meals/rwuyqx1487340229.jpg" },
    { idMeal: "3", strMeal: "Dessert Pudding", strCategory: "Dessert", strMealThumb: "https://www.themealdb.com/images/media/meals/svfvxy1511811220.jpg" },
  ]);

  return (
    <View style={styles.container}>
      <Categories
        categories={categories}
        activeCategory="Beef" // Default active category
        handleChangeCategory={(category) => console.log(category)}
      />
      <Recipe categories={categories} foods={allFood} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 30,
  },
});

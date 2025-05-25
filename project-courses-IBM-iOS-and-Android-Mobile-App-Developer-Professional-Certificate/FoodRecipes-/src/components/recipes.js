import { View, Text, TouchableOpacity, Image, FlatList, StyleSheet } from "react-native";
import React from "react";
import { useNavigation } from "@react-navigation/native";
import { widthPercentageToDP as wp, heightPercentageToDP as hp } from "react-native-responsive-screen";

export default function Recipe({ categories, foods }) {
  const navigation = useNavigation();

  const renderItem = ({ item }) => (
    <ArticleCard item={item} navigation={navigation} />
  );

  return (
    <View style={styles.container} testID="recipesDisplay">
      <FlatList
        data={foods}
        renderItem={renderItem}
        keyExtractor={(item) => item.idMeal.toString()}
        numColumns={2}
        contentContainerStyle={styles.listContainer}
      />
    </View>
  );
}

const ArticleCard = ({ item, navigation }) => {
  return (
    <TouchableOpacity
      style={styles.cardContainer}
      onPress={() => navigation.navigate("RecipeDetail", { recipeId: item.idMeal })}
      testID="articleDisplay"
    >
      <Image source={{ uri: item.strMealThumb }} style={styles.articleImage} />
      <Text style={styles.articleText}>{item.strMeal}</Text>
      <Text style={styles.articleDescription}>{item.strCategory}</Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    marginHorizontal: wp(4),
    marginTop: hp(2),
  },
  cardContainer: {
    flex: 1,
    marginBottom: hp(1.5),
    justifyContent: "center",
  },
  articleImage: {
    width: "100%",
    height: hp(20),
    borderRadius: 35,
    backgroundColor: "rgba(0, 0, 0, 0.05)",
  },
  articleText: {
    fontSize: hp(1.5),
    fontWeight: "600",
    color: "#52525B",
    marginLeft: wp(2),
    marginTop: hp(0.5),
  },
  articleDescription: {
    fontSize: hp(1.2),
    color: "#6B7280",
    marginLeft: wp(2),
    marginTop: hp(0.5),
  },
  listContainer: {
    paddingBottom: hp(3),
  },
});

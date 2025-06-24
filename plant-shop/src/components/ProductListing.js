import React, { useState } from "react";
import { useDispatch } from "react-redux";
import { addToCart } from "../redux/actions";
import ProductCard from "./ProductCard";

function ProductListing() {
  const dispatch = useDispatch();

  const plants = [
    { id: 1, name: "Succulent", price: 15, image: "succulent.jpg" },
    { id: 2, name: "Fern", price: 12, image: "fern.jpg" },
    // Add more plants here
  ];

  const addToCartHandler = (plant) => {
    dispatch(addToCart(plant));
  };

  return (
    <div>
      <h2>Our Products</h2>
      <div className="product-listing">
        {plants.map((plant) => (
          <ProductCard
            key={plant.id}
            plant={plant}
            addToCart={addToCartHandler}
          />
        ))}
      </div>
    </div>
  );
}

export default ProductListing;

import React from "react";

function ProductCard({ plant, addToCart }) {
  return (
    <div className="product">
      <img src={plant.image} alt={plant.name} />
      <h3>{plant.name}</h3>
      <p>${plant.price}</p>
      <button onClick={() => addToCart(plant)}>Add to Cart</button>
    </div>
  );
}

export default ProductCard;

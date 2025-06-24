import React from "react";
import { Link } from "react-router-dom";
import { useSelector } from "react-redux";

function Header() {
  const cartCount = useSelector((state) => state.cart.length);

  return (
    <header>
      <h1>Plant Shop</h1>
      <nav>
        <Link to="/product-listing">Product Listing</Link>
        <Link to="/shopping-cart">Shopping Cart ({cartCount})</Link>
      </nav>
    </header>
  );
}

export default Header;

import React from "react";

function LandingPage() {
  return (
    <div>
      <h2>Welcome to Plant Shop</h2>
      <p>Find your favorite plants here!</p>
      <button onClick={() => window.location.href = '/product-listing'}>
        Get Started
      </button>
    </div>
  );
}

export default LandingPage;

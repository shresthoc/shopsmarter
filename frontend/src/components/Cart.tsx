import React from 'react';
import { useCart } from '../context/CartContext';

const Cart: React.FC = () => {
  const { savedProducts, removeFromCart } = useCart();

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-center mb-8">Saved Products</h1>

      {savedProducts.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-gray-600">No saved products yet.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {savedProducts.map((product, index) => (
            <div key={index} className="bg-white rounded-lg shadow-md overflow-hidden">
            <img
                src={product.image_url}
                alt={product.title}
                className="w-full h-48 object-cover"
            />
              <div className="p-4">
                <h3 className="font-bold text-lg mb-2">{product.title}</h3>
                {product.price && (
                  <p className="text-gray-600 mb-2">${product.price}</p>
                )}
                <div className="flex justify-between items-center">
                  <a
                    href={product.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-500 hover:text-blue-700"
                >
                    View Product
                  </a>
                <button
                    onClick={() => removeFromCart(index)}
                    className="text-red-500 hover:text-red-700"
                >
                    Remove
                </button>
              </div>
            </div>
          </div>
        ))}
        </div>
      )}
    </div>
  );
};

export default Cart;

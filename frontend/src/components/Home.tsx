import React, { useState } from 'react';
import { Product, SearchResponse } from '../types';
import ImageUploader from './ImageUploader';
import ProductCarousel from './ProductCarousel';
import { logger } from '../utils/logger';
import { useCart } from '../context/CartContext';

const Home: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [results, setResults] = useState<Product[]>([]);
  const [loading, setLoading] = useState(false);
  const [prompt, setPrompt] = useState('');
  const { addToCart } = useCart();

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(URL.createObjectURL(file));
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!selectedImage) return;

    setLoading(true);
    try {
      const formData = new FormData();
      const response = await fetch(selectedImage);
      const blob = await response.blob();
      formData.append('image', blob, 'image.jpg');
      if (prompt) {
        formData.append('prompt', prompt);
      }

      console.log('Sending request to backend...');
      const result = await fetch('http://localhost:5001/api/query', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
        },
        body: formData,
      });

      console.log('Response status:', result.status);
      if (!result.ok) {
        const errorText = await result.text();
        console.error('Error response:', errorText);
        throw new Error(`HTTP error! status: ${result.status}, message: ${errorText}`);
      }

      const data: SearchResponse = await result.json();
      console.log('Received response:', data);
      setResults(data.products || []);
    } catch (error) {
      console.error('Error details:', error);
      alert('Error: ' + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleAddToCart = (product: Product) => {
    logger.debug('Adding product to cart', { productId: product.id });
    
    try {
      // Get existing cart
      const cartJson = localStorage.getItem('cart');
      const cart = cartJson ? JSON.parse(cartJson) : [];

      // Check if product already exists
      const existingItem = cart.find((item: any) => item.id === product.id);
      if (existingItem) {
        logger.debug('Updating existing cart item quantity', {
          productId: product.id,
          oldQuantity: existingItem.quantity,
          newQuantity: existingItem.quantity + 1
        });
        existingItem.quantity += 1;
      } else {
        logger.debug('Adding new item to cart', { productId: product.id });
        cart.push({ ...product, quantity: 1 });
      }

      // Save updated cart
      localStorage.setItem('cart', JSON.stringify(cart));
      logger.info('Successfully updated cart', { 
        productId: product.id,
        cartSize: cart.length,
        totalItems: cart.reduce((sum: number, item: any) => sum + item.quantity, 0)
      });

      // Show success message
      alert('Added to cart!');
    } catch (err) {
      logger.error('Error updating cart', err);
      alert('Failed to add item to cart. Please try again.');
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-center mb-8">Find Similar Products</h1>
      
      <form onSubmit={handleSubmit} className="bg-white shadow-md rounded-lg p-6 mb-8">
        <div className="mb-6">
          <label className="block text-gray-700 text-sm font-bold mb-2">
            Upload an Image
          </label>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="w-full p-2 border rounded"
          />
          {selectedImage && (
            <img
              src={selectedImage}
              alt="Selected"
              className="mt-4 max-w-xs mx-auto rounded shadow-lg"
            />
          )}
        </div>

        <div className="mb-6">
          <label className="block text-gray-700 text-sm font-bold mb-2">
            Add Description (Optional)
          </label>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Describe what you're looking for..."
            className="w-full p-2 border rounded"
          />
        </div>

        <button
          type="submit"
          disabled={!selectedImage || loading}
          className={`w-full py-2 px-4 rounded font-bold ${
            !selectedImage || loading
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-500 hover:bg-blue-700 text-white'
          }`}
        >
          {loading ? 'Searching...' : 'Find Similar Products'}
        </button>
      </form>

      {results.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {results.map((product, index) => (
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
                    onClick={() => addToCart(product)}
                    className="text-green-500 hover:text-green-700"
                  >
                    Save
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

export default Home; 
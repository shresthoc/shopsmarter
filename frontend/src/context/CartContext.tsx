import React, { createContext, useContext, useState, useEffect } from 'react';
import { Product } from '../types';

interface CartContextType {
  savedProducts: Product[];
  addToCart: (product: Product) => void;
  removeFromCart: (index: number) => void;
  clearCart: () => void;
}

const CartContext = createContext<CartContextType | undefined>(undefined);

export const CartProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [savedProducts, setSavedProducts] = useState<Product[]>(() => {
    const saved = localStorage.getItem('cart');
    return saved ? JSON.parse(saved) : [];
  });

  useEffect(() => {
    localStorage.setItem('cart', JSON.stringify(savedProducts));
  }, [savedProducts]);

  const addToCart = (product: Product) => {
    setSavedProducts(prev => [...prev, product]);
  };

  const removeFromCart = (index: number) => {
    setSavedProducts(prev => prev.filter((_, i) => i !== index));
  };

  const clearCart = () => {
    setSavedProducts([]);
  };

  return (
    <CartContext.Provider value={{ savedProducts, addToCart, removeFromCart, clearCart }}>
      {children}
    </CartContext.Provider>
  );
};

export const useCart = () => {
  const context = useContext(CartContext);
  if (context === undefined) {
    throw new Error('useCart must be used within a CartProvider');
  }
  return context;
}; 
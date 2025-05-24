import React from 'react';
import Slider from 'react-slick';
import { ShoppingCartIcon } from '@heroicons/react/24/outline';
import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';

interface Product {
  id: string;
  title: string;
  price: number;
  image_url: string;
  url: string;
  source: string;
  similarity_score: number;
}

interface ProductCarouselProps {
  products: Product[];
  onAddToCart: (product: Product) => void;
}

const ProductCarousel: React.FC<ProductCarouselProps> = ({ products, onAddToCart }) => {
  const settings = {
    dots: true,
    infinite: false,
    speed: 500,
    slidesToShow: 4,
    slidesToScroll: 4,
    responsive: [
      {
        breakpoint: 1024,
        settings: {
          slidesToShow: 3,
          slidesToScroll: 3,
        }
      },
      {
        breakpoint: 768,
        settings: {
          slidesToShow: 2,
          slidesToScroll: 2,
        }
      },
      {
        breakpoint: 640,
        settings: {
          slidesToShow: 1,
          slidesToScroll: 1,
        }
      }
    ]
  };

  const handleProductClick = (url: string) => {
    window.open(url, '_blank');
  };

  return (
    <div className="w-full">
      <Slider {...settings}>
        {products.map((product) => (
          <div key={product.id} className="px-2">
            <div 
              className="bg-white rounded-lg shadow-md overflow-hidden cursor-pointer hover:shadow-lg transition-shadow"
              onClick={() => handleProductClick(product.url)}
            >
              <div className="aspect-square relative">
                <img
                  src={product.image_url}
                  alt={product.title}
                  className="w-full h-full object-cover"
                />
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-4">
                  <div className="text-white text-sm font-medium truncate">
                    {product.title}
                  </div>
                </div>
              </div>
              
              <div className="p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="text-gray-900 font-bold">
                    ${product.price.toFixed(2)}
                  </div>
                  <div className="text-gray-500 text-sm">
                    from {product.source}
                  </div>
                </div>
                
                <div className="flex space-x-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onAddToCart(product);
                    }}
                    className="flex-1 bg-blue-600 text-white px-4 py-2 rounded-md 
                      hover:bg-blue-700 transition-colors flex items-center justify-center space-x-2"
                  >
                    <ShoppingCartIcon className="h-5 w-5" />
                    <span>Add to Cart</span>
                  </button>
                </div>
                
                <div className="text-xs text-gray-500">
                  Similarity Score: {(1 - product.similarity_score).toFixed(2)}
                </div>
              </div>
            </div>
          </div>
        ))}
      </Slider>
    </div>
  );
};

export default ProductCarousel; 
export interface Product {
  id: string;
  title: string;
  price: number;
  image_url: string;
  url: string;
}

export interface SearchResponse {
  products: Product[];
} 
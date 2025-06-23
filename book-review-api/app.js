const express = require('express');
const axios = require('axios');
const app = express();
app.use(express.json());  // Untuk membaca body JSON

// Endpoint untuk mendapatkan daftar buku
app.get('/books', async (req, res) => {
  try {
    // Menggunakan Google Books API untuk mengambil daftar buku berdasarkan subjek
    const response = await axios.get('https://www.googleapis.com/books/v1/volumes?q=subject:fiction');
    res.json(response.data.items); // Mengirimkan daftar buku dalam format JSON
  } catch (error) {
    res.status(500).send('Error retrieving books');
  }
});

// Endpoint untuk mendapatkan buku berdasarkan ISBN
app.get('/books/:isbn', async (req, res) => {
  const isbn = req.params.isbn;
  try {
    // Menggunakan Google Books API untuk mendapatkan buku berdasarkan ISBN
    const response = await axios.get(`https://www.googleapis.com/books/v1/volumes?q=isbn:${isbn}`);
    res.json(response.data.items); // Mengirimkan detail buku berdasarkan ISBN
  } catch (error) {
    res.status(500).send('Error retrieving book by ISBN');
  }
});

// Endpoint untuk mendapatkan buku berdasarkan judul
app.get('/books/title/:title', async (req, res) => {
  const title = req.params.title;
  try {
    // Menggunakan Google Books API untuk mendapatkan buku berdasarkan judul
    const response = await axios.get(`https://www.googleapis.com/books/v1/volumes?q=intitle:${title}`);
    res.json(response.data.items); // Mengirimkan daftar buku berdasarkan judul
  } catch (error) {
    res.status(500).send('Error retrieving books by title');
  }
});

// Endpoint untuk mendapatkan buku berdasarkan penulis
app.get('/books/author/:author', async (req, res) => {
  const author = req.params.author;
  try {
    // Menggunakan Google Books API untuk mendapatkan buku berdasarkan penulis
    const response = await axios.get(`https://www.googleapis.com/books/v1/volumes?q=inauthor:${author}`);
    res.json(response.data.items); // Mengirimkan daftar buku berdasarkan penulis
  } catch (error) {
    res.status(500).send('Error retrieving books by author');
  }
});

// Endpoint untuk mendapatkan review buku
app.get('/reviews/:bookIsbn', async (req, res) => {
  const bookIsbn = req.params.bookIsbn;
  try {
    // Ganti URL dengan API review yang sesuai
    const response = await axios.get(`https://api.example.com/reviews?isbn=${bookIsbn}`);
    res.json(response.data);
  } catch (error) {
    res.status(500).send('Error retrieving reviews');
  }
});

// Endpoint untuk menambahkan review buku
app.post('/reviews', async (req, res) => {
  const { userId, bookIsbn, review } = req.body; // Menerima data userId, bookIsbn, dan review dari body request
  try {
    // Ganti URL dengan API review yang sesuai
    const response = await axios.post('https://api.example.com/reviews', { userId, bookIsbn, review });
    res.status(200).send('Review added');
  } catch (error) {
    res.status(500).send('Error adding review');
  }
});

// Endpoint untuk menghapus review buku
app.delete('/reviews/:reviewId', async (req, res) => {
  const reviewId = req.params.reviewId; // Mengambil reviewId dari parameter URL
  try {
    // Ganti URL dengan API review yang sesuai
    const response = await axios.delete(`https://api.example.com/reviews/${reviewId}`);
    res.status(200).send('Review deleted');
  } catch (error) {
    res.status(500).send('Error deleting review');
  }
});

// Endpoint untuk menambahkan user baru
app.post('/register', async (req, res) => {
  const { username, password } = req.body;
  try {
    // Ganti URL dengan API registrasi yang sesuai
    const response = await axios.post('https://api.example.com/register', { username, password });
    res.status(200).send('User registered');
  } catch (error) {
    res.status(500).send('Error registering user');
  }
});

// Endpoint untuk login user
app.post('/login', async (req, res) => {
  const { username, password } = req.body;
  try {
    // Ganti URL dengan API login yang sesuai
    const response = await axios.post('https://api.example.com/login', { username, password });
    res.status(200).send('User logged in');
  } catch (error) {
    res.status(500).send('Error logging in');
  }
});

// Endpoint untuk mendapatkan semua buku menggunakan callback function
app.get('/books/callback', (req, res) => {
  axios.get('https://www.googleapis.com/books/v1/volumes?q=subject:fiction')
    .then(response => {
      res.json(response.data.items);
    })
    .catch(error => {
      res.status(500).send('Error retrieving books');
    });
});

// Menjalankan server di port 3000
app.listen(3000, () => {
  console.log('Server running on http://localhost:3000');
});

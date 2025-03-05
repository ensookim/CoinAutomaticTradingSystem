import React, { Component } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Login from './components/Login';
// import Register from './components/Register';
import Main_Page from './components/Chart';
import './components/CSS/App.css';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isAuthenticated: false
    };
  }

  authenticate = () => {
    this.setState({ isAuthenticated: true });
  };

  render() {
    return (
      <Router>
        <Routes>
          <Route path="/login" element={<Login authenticate={this.authenticate} />} />
          {/* <Route path="/register" element={<Register />} /> */}
          <Route
            path="/chart"
            element={this.state.isAuthenticated ? <Main_Page /> : <Navigate to="/login" />}
          />
          <Route path="/" element={<Navigate to="/login" />} />
        </Routes>
      </Router>
    );
  }
}

export default App;

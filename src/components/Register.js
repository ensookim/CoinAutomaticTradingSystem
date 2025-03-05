import React, { Component } from 'react';
import axios from 'axios';

class Register extends Component {
  state = {
    username: '',
    password: '',
    error: ''
  };

  handleChange = (e) => {
    this.setState({ [e.target.name]: e.target.value });
  };

  handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/register', {
        username: this.state.username,
        password: this.state.password
      });
      if (response.status === 201) {
        this.props.history.push('/login');
      }
    } catch (error) {
      this.setState({ error: 'Error registering user' });
    }
  };

  render() {
    return (
      <div>
        <h2>Register</h2>
        <form onSubmit={this.handleSubmit}>
          <div>
            <label>Username</label>
            <input type="text" name="username" onChange={this.handleChange} />
          </div>
          <div>
            <label>Password</label>
            <input type="password" name="password" onChange={this.handleChange} />
          </div>
          <button type="submit">Register</button>
          {this.state.error && <p>{this.state.error}</p>}
        </form>
      </div>
    );
  }
}

export default Register;

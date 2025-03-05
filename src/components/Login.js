import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import styles from './CSS/styles.module.css'
//import styles from './components/CSS/styles.module.css'; 

const Login = ({ authenticate }) => {
  const containerRef = useRef(null); 
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isSignUp, setIsSignUp] = useState(false); 
  const navigate = useNavigate();

  useEffect(() => {
    if (containerRef.current) {
      setTimeout(() => {
        containerRef.current.classList.add(styles['sign-in']);
      }, 200);
    }
  }, []);

  useEffect(() => {
    if (containerRef.current) {
      if (isSignUp) {
        containerRef.current.classList.add(styles['sign-up']);
        containerRef.current.classList.remove(styles['sign-in']);
      } else {
        containerRef.current.classList.add(styles['sign-in']);
        containerRef.current.classList.remove(styles['sign-up']);
      }
    }
  }, [isSignUp]);

  const toggle = () => {
    setIsSignUp(prevState => !prevState); 
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    if (name === 'username') setUsername(value);
    else if (name === 'password') setPassword(value);
    else if (name === 'email') setEmail(value);
    else if (name === 'confirmPassword') setConfirmPassword(value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // 이메일 형식 검사 정규 표현식
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    
    if (isSignUp) {
      // 이메일 형식 유효성 검사
      if (!emailRegex.test(email)) {
        alert('유효한 이메일 형식이 아닙니다.'); // Alert 추가
        return;
      }
      
      if (password !== confirmPassword) {
        alert('비밀번호가 일치하지 않습니다.'); // Alert 추가
        return;
      }
      
      try {
        const response = await axios.post('http://localhost:5000/register', {
          username,
          email,
          password,
        });
        if (response.status === 200) {
          alert('회원가입 성공! 로그인 화면으로 이동합니다.'); // 성공 알림
          setTimeout(() => {
            setIsSignUp(false); // 로그인 화면으로 전환
          }, 2000); // 2초 후에 로그인 화면으로 전환
        }
      } catch (error) {
        alert('계정 생성 중 오류가 발생했습니다.'); // Alert 추가
      }
    } else {
      try {
        const response = await axios.post('http://localhost:5000/login', {
          username,
          password,
        });
        if (response.status === 200) {
          authenticate();
          console.log('Request data:', response.data);
          navigate('/chart', { state: { username } });
        }
      } catch (error) {
        alert('잘못된 자격 증명입니다.'); // Alert 추가
      }
    }
  };
  
  

  return (
    <div id="container" className={styles.container} ref={containerRef}>
      <div className={styles.row}>
        {/* SIGN UP */}
        <div className={`${styles.col} ${styles['align-items-center']} ${styles['flex-col']} ${styles['sign-up']}`}>
          <div className={`${styles['form-wrapper']} ${styles['align-items-center']}`}>
            <div className={`${styles.form} ${styles['sign-up']}`}>
              <div className={styles['input-group']}>
                <i className='bx bxs-user'></i>
                <input
                  type="text"
                  placeholder="Username"
                  name="username"
                  value={username}
                  onChange={handleChange}
                />
              </div>
              <div className={styles['input-group']}>
                <i className='bx bx-mail-send'></i>
                <input
                  type="email"
                  placeholder="Email"
                  name="email"
                  value={email}
                  onChange={handleChange}
                />
              </div>
              <div className={styles['input-group']}>
                <i className='bx bxs-lock-alt'></i>
                <input
                  type="password"
                  placeholder="Password"
                  name="password"
                  value={password}
                  onChange={handleChange}
                />
              </div>
              <div className={styles['input-group']}>
                <i className='bx bxs-lock-alt'></i>
                <input
                  type="password"
                  placeholder="Confirm password"
                  name="confirmPassword"
                  value={confirmPassword}
                  onChange={handleChange}
                />
              </div>
              <button onClick={handleSubmit}>
                Sign up
              </button>
              <p>
                <span>
                  Already have an account?
                </span>
                <b onClick={toggle} className="pointer">
                  Sign in here
                </b>
              </p>
            </div>
          </div>
        </div>
        {/* END SIGN UP */}
        {/* SIGN IN */}
        <div className={`${styles.col} ${styles['align-items-center']} ${styles['flex-col']} ${styles['sign-in']}`}>
          <div className={`${styles['form-wrapper']} ${styles['align-items-center']}`}>
            <div className={`${styles.form} ${styles['sign-in']}`}>
              <div className={styles['input-group']}>
                <i className='bx bxs-user'></i>
                <input
                  type="text"
                  placeholder="Username"
                  name="username"
                  value={username}
                  onChange={handleChange}
                />
              </div>
              <div className={styles['input-group']}>
                <i className='bx bxs-lock-alt'></i>
                <input
                  type="password"
                  placeholder="Password"
                  name="password"
                  value={password}
                  onChange={handleChange}
                />
              </div>
              <button onClick={handleSubmit}>
                Sign in
              </button>
              <p>
                <b>
                  Forgot password?
                </b>
              </p>
              <p>
                <span>
                  Don't have an account? -  
                </span>
                <b onClick={toggle} className="pointer">
                  Sign up here
                </b>
              </p>
            </div>
          </div>
        </div>
        {/* END SIGN IN */}
      </div>
      {/* CONTENT SECTION */}
      <div className={`${styles.row} ${styles['content-row']}`}>
        {/* SIGN IN CONTENT */}
        <div className={`${styles.col} ${styles['align-items-center']} ${styles['flex-col']}`}>
          <div className={`${styles.text} ${styles['sign-in']}`}>
            <h2>
              Welcome
            </h2>
          </div>
          <div className={`${styles.img} ${styles['sign-in']}`}>
          </div>
        </div>
        {/* END SIGN IN CONTENT */}
        {/* SIGN UP CONTENT */}
        <div className={`${styles.col} ${styles['align-items-center']} ${styles['flex-col']}`}>
          <div className={`${styles.img} ${styles['sign-up']}`}>
          </div>
          <div className={`${styles.text} ${styles['sign-up']}`}>
            <h2>
              Join with us
            </h2>
          </div>
        </div>
        {/* END SIGN UP CONTENT */}
      </div>
    </div>
  );
};

export default Login;

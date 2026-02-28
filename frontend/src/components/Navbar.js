// src/components/Navbar.js
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../AuthContext';
import { LogOut, Home, Calculator, UploadCloud } from 'lucide-react';
import styles from './Navbar.module.css';

export default function Navbar() {
  const { user, signout } = useAuth();
  const navigate = useNavigate();
  const { pathname } = useLocation();

  const links = [
    { to:'/',          label:'Home',       icon:<Home size={15}/> },
    { to:'/calculate', label:'Score',      icon:<Calculator size={15}/> },
    { to:'/batch',     label:'Batch',      icon:<UploadCloud size={15}/> },
  ];

  return (
    <nav className={styles.nav}>
      <div className={styles.inner}>
        <button className={styles.brand} onClick={()=>navigate('/')}>
          <div className={styles.logo}>Z</div>
          <span>ZENITH</span>
        </button>
        <div className={styles.links}>
          {links.map(l=>(
            <button key={l.to}
              className={`${styles.link} ${pathname===l.to?styles.active:''}`}
              onClick={()=>navigate(l.to)}>
              {l.icon}{l.label}
            </button>
          ))}
        </div>
        <div className={styles.right}>
          <div className={styles.user}>
            <div className={styles.avatar}>{user?.name?.[0]??'U'}</div>
            <div className={styles.userInfo}>
              <span className={styles.userName}>{user?.name}</span>
              <span className={styles.userRole}>{user?.role}</span>
            </div>
          </div>
          <button className={styles.logout} onClick={()=>{ signout(); navigate('/login'); }}>
            <LogOut size={16}/>
          </button>
        </div>
      </div>
    </nav>
  );
}

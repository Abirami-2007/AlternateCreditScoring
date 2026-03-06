// src/components/Navbar.js
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../AuthContext';
import { LogOut, Home, Calculator, UploadCloud } from 'lucide-react';
import { motion } from 'framer-motion';
import styles from './Navbar.module.css';

export default function Navbar() {
  const { user, signout } = useAuth();
  const navigate = useNavigate();
  const { pathname } = useLocation();

  const links = [
    { to: '/', label: 'Home', icon: <Home size={16} /> },
    { to: '/calculate', label: 'Score', icon: <Calculator size={16} /> },
    { to: '/batch', label: 'Batch', icon: <UploadCloud size={16} /> },
  ];

  return (
    <motion.nav
      className={styles.nav}
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
    >
      <div className={styles.inner}>
        <button className={styles.brand} onClick={() => navigate('/')}>
          <motion.div
            className={styles.logo}
            whileHover={{ rotate: 180, scale: 1.1 }}
            transition={{ duration: 0.3 }}
          >
          F
          </motion.div>
          <span>FINOVA</span>
        </button>
        <div className={styles.links}>
          {links.map((l, i) => (
            <motion.button
              key={l.to}
              className={`${styles.link} ${pathname === l.to ? styles.active : ''}`}
              onClick={() => navigate(l.to)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + i * 0.1 }}
            >
              {l.icon}
              <span className={styles.linkLabel}>{l.label}</span>
              {pathname === l.to && (
                <motion.div
                  className={styles.activeIndicator}
                  layoutId="activeTab"
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
            </motion.button>
          ))}
        </div>
        <div className={styles.right}>
          <motion.div
            className={styles.user}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            <div className={styles.avatar}>{user?.name?.[0]?.toUpperCase() ?? 'U'}</div>
            <div className={styles.userInfo}>
              <span className={styles.userName}>{user?.name || 'User'}</span>
              <span className={styles.userRole}>{user?.role || 'Guest'}</span>
            </div>
          </motion.div>
          <motion.button
            className={styles.logout}
            onClick={() => { signout(); navigate('/login'); }}
            whileHover={{ scale: 1.1, backgroundColor: 'rgba(239, 68, 68, 0.1)', color: '#EF4444', borderColor: '#EF4444' }}
            whileTap={{ scale: 0.9 }}
          >
            <LogOut size={16} />
          </motion.button>
        </div>
      </div>
    </motion.nav>
  );
}

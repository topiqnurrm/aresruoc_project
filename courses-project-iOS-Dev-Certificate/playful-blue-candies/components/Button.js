import { Pressable, StyleSheet, Text } from 'react-native';
import { COLORS } from '../constants/theme';
import useFontSize from '../hooks/useFontSize';

export const Button = ({ children, style, disabled, ...otherProps }) => {
  const fontStyle = useFontSize(18);

  return (
    <Pressable
      hitSlop={10}
      style={[
        styles.btnContainer,
        disabled
          ? styles.disabledbtnContainerStyles
          : styles.enabledbtnContainerStyles,
        style,
      ]}
      disabled={disabled}
      {...otherProps}>
      <Text style={[styles.btnText, fontStyle]}>{children}</Text>
    </Pressable>
  );
};

const styles = StyleSheet.create({
  btnContainer: {
    backgroundColor: COLORS.darkGray,
    borderRadius: 8,
    padding: 8,
  },
  disabledbtnContainerStyles: {
    cursor: 'not-allowed', // for web, no use on mobile
    opacity: 0.5,
  },
  enabledbtnContainerStyles: {
    cursor: 'pointer',
  },
  btnText: {
    color: COLORS.white,
    textAlign: 'center',
  },
});

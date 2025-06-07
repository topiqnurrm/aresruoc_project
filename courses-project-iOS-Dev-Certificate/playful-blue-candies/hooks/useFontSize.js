import { useWindowDimensions } from 'react-native';

const useFontSize = (size = 16) => {
  const { fontScale } = useWindowDimensions();
  const fontSize = size * fontScale;
  const multiplier = fontSize > 80 ? 1.2 : 1.6;
  return {
    fontSize,
    lineHeight: fontSize * multiplier,
  };
};

export default useFontSize;

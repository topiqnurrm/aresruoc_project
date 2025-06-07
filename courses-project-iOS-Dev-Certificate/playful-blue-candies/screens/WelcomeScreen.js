import { Image, StyleSheet, Text, View } from 'react-native';
import { Button } from '../components/Button';
import { BaseText } from '../components/Typography';
import { BASE_SCREEN_STYLE } from '../constants/theme';
import { ROUTES } from '../navigators/RootNavigator';

const SiteInfo = () => (
  <View style={styles.siteInfoContainer}>
    <Image
      source={require('../assets/little-lemon-logo.png')}
      style={styles.logo}
    />
    <BaseText>Little Lemon, your local mediterranean bistro</BaseText>
  </View>
);

const WelcomeScreen = ({ navigation }) => {
  return (
    <View style={styles.screenContainer}>
      <SiteInfo />
      <Button
        onPress={() => navigation.navigate(ROUTES.newsletter)}
        style={styles.button}>
        Subscribe to our NewsLetter!
      </Button>
    </View>
  );
};

export default WelcomeScreen;

const styles = StyleSheet.create({
  screenContainer: BASE_SCREEN_STYLE,
  siteInfoContainer: {
    flex: 1,
    justifyContent: 'center',
    gap: 24,
  },
  logo: {
    resizeMode: 'contain',
    width: 'auto',
    height: 200,
  },
  button: {
    marginBlock: 24,
  },
});
